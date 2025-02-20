from tensorflow.keras.layers import Flatten as batch_flatten

from btgym_tf2.algorithms.policy.base import BaseAacPolicy
from btgym_tf2.algorithms.nn.networks import *
from btgym_tf2.algorithms.utils import *

from btgym_tf2.spaces import DictSpace, ActionDictSpace


class StackedLstmPolicy(BaseAacPolicy):
    """
    Conv.-Stacked_LSTM policy, based on `NAV A3C agent` architecture from

    `LEARNING TO NAVIGATE IN COMPLEX ENVIRONMENTS` by Mirowski et all. and

    `LEARNING TO REINFORCEMENT LEARN` by JX Wang et all.

    Papers:

    https://arxiv.org/pdf/1611.03673.pdf

    https://arxiv.org/pdf/1611.05763.pdf
    """

    def __init__(self,
                 ob_space,
                 ac_space,
                 rp_sequence_size,
                 state_encoder_class_ref=conv_2d_network,
                 lstm_class_ref=tf.compat.v1.nn.rnn_cell.LSTMCell,
                 lstm_layers=(256, 256),
                 linear_layer_ref=noisy_linear,
                 share_encoder_params=False,
                 dropout_keep_prob=1.0,
                 action_dp_alpha=200.0,
                 aux_estimate=False,
                 encode_internal_state=None,
                 static_rnn=True,
                 shared_p_v=False,
                 **kwargs):
        """
        Defines [partially shared] on/off-policy networks for estimating  action-logits, value function,
        reward and state 'pixel_change' predictions.
        Expects multi-modal observation as array of shape `ob_space`.

        Args:
            ob_space:               instance of btgym_tf2.spaces.DictSpace
            ac_space:               instance of btgym_tf2.spaces.ActionDictSpace
            rp_sequence_size:       reward prediction sample length
            lstm_class_ref:         tf.nn.lstm class to use
            lstm_layers:            tuple of LSTM layers sizes
            linear_layer_ref:       linear layer class to use
            share_encoder_params:   bool, whether to share encoder parameters for every 'external' data stream
            dropout_keep_prob:      in (0, 1] dropout regularisation parameter
            action_dp_alpha:
            aux_estimate:           (bool), if True - add auxiliary tasks estimations to self.callbacks dictionary
            encode_internal_state:  legacy, not used
            static_rnn:             (bool), it True - use static rnn graph, dynamic otherwise
            **kwargs                not used
        """

        assert isinstance(ob_space, DictSpace), \
            'Expected observation space be instance of btgym_tf2.spaces.DictSpace, got: {}'.format(ob_space)
        self.ob_space = ob_space

        assert isinstance(ac_space, ActionDictSpace), \
            'Expected action space be instance of btgym_tf2.spaces.ActionDictSpace, got: {}'.format(ac_space)

        self.ac_space = ac_space
        self.rp_sequence_size = rp_sequence_size
        self.state_encoder_class_ref = state_encoder_class_ref
        self.lstm_class = lstm_class_ref
        self.lstm_layers = lstm_layers
        self.action_dp_alpha = action_dp_alpha
        self.aux_estimate = aux_estimate
        self.callback = {}
        # self.encode_internal_state = encode_internal_state
        self.share_encoder_params = share_encoder_params
        if self.share_encoder_params:
            self.reuse_encoder_params = tf.compat.v1.AUTO_REUSE

        else:
            self.reuse_encoder_params = False
        self.static_rnn = static_rnn
        self.dropout_keep_prob = dropout_keep_prob
        assert 0 < self.dropout_keep_prob <= 1, 'Dropout keep_prob value should be in (0, 1]'

        self.debug = {}

        # Placeholders for obs. state input:
        self.on_state_in = nested_placeholders(self.ob_space._shape, batch_dim=None, name='on_policy_state_in')
        self.off_state_in = nested_placeholders(self.ob_space._shape, batch_dim=None, name='off_policy_state_in_pl')
        self.rp_state_in = nested_placeholders(self.ob_space._shape, batch_dim=None, name='rp_state_in')

        # Placeholders for previous step action[multi-categorical vector encoding]  and reward [scalar]:
        self.on_last_a_in = tf.compat.v1.placeholder(
            tf.float32,
            [None, self.ac_space.encoded_depth],
            name='on_policy_last_action_in_pl'
        )
        self.on_last_reward_in = tf.compat.v1.placeholder(tf.float32, [None], name='on_policy_last_reward_in_pl')

        self.off_last_a_in = tf.compat.v1.placeholder(
            tf.float32,
            [None, self.ac_space.encoded_depth],
            name='off_policy_last_action_in_pl'
        )
        self.off_last_reward_in = tf.compat.v1.placeholder(tf.float32, [None], name='off_policy_last_reward_in_pl')

        # Placeholders for rnn batch and time-step dimensions:
        self.on_batch_size = tf.compat.v1.placeholder(tf.int32, name='on_policy_batch_size')
        self.on_time_length = tf.compat.v1.placeholder(tf.int32, name='on_policy_sequence_size')

        self.off_batch_size = tf.compat.v1.placeholder(tf.int32, name='off_policy_batch_size')
        self.off_time_length = tf.compat.v1.placeholder(tf.int32, name='off_policy_sequence_size')

        self.debug['on_state_in_keys'] = list(self.on_state_in.keys())

        # Dropout related:
        try:
            if self.train_phase is not None:
                pass

        except AttributeError:
            self.train_phase = tf.compat.v1.placeholder_with_default(
                tf.constant(False, dtype=tf.bool),
                shape=(),
                name='train_phase_flag_pl'
            )
        self.keep_prob = 1.0 - (1.0 - self.dropout_keep_prob) * tf.cast(self.train_phase, tf.float32)

        # Default parameters:
        default_kwargs = dict(
            conv_2d_filter_size=[3, 1],
            conv_2d_stride=[2, 1],
            conv_2d_num_filters=[32, 32, 64, 64],
            pc_estimator_stride=[2, 1],
            duell_pc_x_inner_shape=(6, 1, 32),  # [6,3,32] if swapping W-C dims
            duell_pc_filter_size=(4, 1),
            duell_pc_stride=(2, 1),
            keep_prob=self.keep_prob,
        )
        # Insert if not already:
        for key, default_value in default_kwargs.items():
            if key not in kwargs.keys():
                kwargs[key] = default_value

        # Base on-policy AAC network:
        self.modes_to_encode = ['external', 'internal']
        for mode in self.modes_to_encode:
            assert mode in self.on_state_in.keys(), \
                'Required top-level mode `{}` not found in state shape specification'.format(mode)

        # Separately encode than concatenate all `external` and 'internal' states modes,
        # [jointly] encode every stream within mode:
        self.on_aac_x_encoded = {}
        for key in self.modes_to_encode:
            if isinstance(self.on_state_in[key], dict):  # got dictionary of data streams
                if self.share_encoder_params:
                    layer_name_template = 'encoded_{}_shared'
                else:
                    layer_name_template = 'encoded_{}_{}'
                encoded_streams = {
                    name: tf.compat.v1.layers.flatten(
                        self.state_encoder_class_ref(
                            x=stream,
                            ob_space=self.ob_space._shape[key][name],
                            ac_space=self.ac_space,
                            name=layer_name_template.format(key, name),
                            reuse=self.reuse_encoder_params,  # shared params for all streams in mode
                            **kwargs
                        )
                    )
                    for name, stream in self.on_state_in[key].items()
                }
                encoded_mode = tf.concat(
                    list(encoded_streams.values()),
                    axis=-1,
                    name='multi_encoded_{}'.format(key)
                )
            else:
                # Got single data stream:
                encoded_mode = tf.compat.v1.layers.flatten(
                    self.state_encoder_class_ref(
                        x=self.on_state_in[key],
                        ob_space=self.ob_space._shape[key],
                        ac_space=self.ac_space,
                        name='encoded_{}'.format(key),
                        **kwargs
                    )
                )
            self.on_aac_x_encoded[key] = encoded_mode

        self.debug['on_state_external_encoded_dict'] = self.on_aac_x_encoded

        # on_aac_x = tf.concat(list(self.on_aac_x_encoded.values()), axis=-1, name='on_state_external_encoded')
        on_aac_x = self.on_aac_x_encoded['external']

        self.debug['on_state_external_encoded'] = on_aac_x

        # TODO: for encoder prediction test, output `naive` estimates for logits and value directly from encoder:
        [self.on_simple_logits, self.on_simple_value, _] = dense_aac_network(
            tf.compat.v1.layers.flatten(on_aac_x),
            ac_space_depth=self.ac_space.one_hot_depth,
            linear_layer_ref=linear_layer_ref,
            name='aac_dense_simple_pi_v'
        )

        # Reshape rnn inputs for batch training as: [rnn_batch_dim, rnn_time_dim, flattened_depth]:
        x_shape_dynamic = tf.shape(input=on_aac_x)
        max_seq_len = tf.cast(x_shape_dynamic[0] / self.on_batch_size, tf.int32)
        x_shape_static = on_aac_x.get_shape().as_list()

        on_last_action_in = tf.reshape(
            self.on_last_a_in,
            [self.on_batch_size, max_seq_len, self.ac_space.encoded_depth]
        )
        on_last_r_in = tf.reshape(self.on_last_reward_in, [self.on_batch_size, max_seq_len, 1])

        on_aac_x = tf.reshape(on_aac_x, [self.on_batch_size, max_seq_len, np.prod(x_shape_static[1:])])

        # # Prepare `internal` state, if any:
        # if 'internal' in list(self.on_state_in.keys()):
        #     if self.encode_internal_state:
        #         # Use convolution encoder:
        #         on_x_internal = self.state_encoder_class_ref(
        #             x=self.on_state_in['internal'],
        #             ob_space=self.ob_space._shape['internal'],
        #             ac_space=self.ac_space,
        #             name='encoded_internal',
        #             **kwargs
        #         )
        #         x_int_shape_static = on_x_internal.get_shape().as_list()
        #         on_x_internal = [
        #             tf.reshape(on_x_internal, [self.on_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])])]
        #         self.debug['on_state_internal_encoded'] = on_x_internal
        #
        #     else:
        #         # Feed as is:
        #         x_int_shape_static = self.on_state_in['internal'].get_shape().as_list()
        #         on_x_internal = tf.reshape(
        #             self.on_state_in['internal'],
        #             [self.on_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])]
        #         )
        #         self.debug['on_state_internal_encoded'] = on_x_internal
        #         on_x_internal = [on_x_internal]
        #
        # else:
        #     on_x_internal = []

        on_x_internal = self.on_aac_x_encoded['internal']

        # Reshape to batch-feed rnn:
        x_int_shape_static = on_x_internal.get_shape().as_list()
        on_x_internal = tf.reshape(
            on_x_internal,
            [self.on_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])]
        )
        self.debug['on_state_internal_encoded'] = on_x_internal

        on_x_internal = [on_x_internal]

        # Prepare datetime index if any:
        if 'datetime' in list(self.on_state_in.keys()):
            x_dt_shape_static = self.on_state_in['datetime'].get_shape().as_list()
            on_x_dt = tf.reshape(
                self.on_state_in['datetime'],
                [self.on_batch_size, max_seq_len, np.prod(x_dt_shape_static[1:])]
            )
            on_x_dt = [on_x_dt]

        else:
            on_x_dt = []

        self.debug['on_state_dt_encoded'] = on_x_dt
        self.debug['conv_input_to_lstm1'] = on_aac_x

        # Feed last last_reward into LSTM_1 layer along with encoded `external` state features and datetime stamp:
        # on_stage2_1_input = [on_aac_x, on_last_action_in, on_last_reward_in] + on_x_dt
        on_stage2_1_input = [on_aac_x, on_last_r_in] #+ on_x_dt

        # Feed last_action, encoded `external` state,  `internal` state, datetime stamp into LSTM_2:
        # on_stage2_2_input = [on_aac_x, on_last_action_in, on_last_reward_in] + on_x_internal + on_x_dt
        on_stage2_2_input = [on_aac_x, on_last_action_in] + on_x_internal #+ on_x_dt

        # LSTM_1 full input:
        on_aac_x = tf.concat(on_stage2_1_input, axis=-1)

        self.debug['concat_input_to_lstm1'] = on_aac_x

        # First LSTM layer takes encoded `external` state:
        [on_x_lstm_1_out, self.on_lstm_1_init_state, self.on_lstm_1_state_out, self.on_lstm_1_state_pl_flatten] =\
            lstm_network(
                x=on_aac_x,
                lstm_sequence_length=self.on_time_length,
                lstm_class=lstm_class_ref,
                lstm_layers=(lstm_layers[0],),
                static=static_rnn,
                name='lstm_1',
                **kwargs,
            )

        # var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        # print('var_list: ', var_list)

        self.debug['on_x_lstm_1_out'] = on_x_lstm_1_out
        self.debug['self.on_lstm_1_state_out'] = self.on_lstm_1_state_out
        self.debug['self.on_lstm_1_state_pl_flatten'] = self.on_lstm_1_state_pl_flatten

        # For time_flat only: Reshape on_lstm_1_state_out from [1,2,20,size] -->[20,1,2,size] --> [20,1, 2xsize]:
        reshape_lstm_1_state_out = tf.transpose(a=self.on_lstm_1_state_out, perm=[2, 0, 1, 3])
        reshape_lstm_1_state_out_shape_static = reshape_lstm_1_state_out.get_shape().as_list()

        # Take policy logits off first LSTM-dense layer:
        # Reshape back to [batch, flattened_depth], where batch = rnn_batch_dim * rnn_time_dim:
        x_shape_static = on_x_lstm_1_out.get_shape().as_list()
        rsh_on_x_lstm_1_out = tf.reshape(on_x_lstm_1_out, [x_shape_dynamic[0], x_shape_static[-1]])

        self.debug['reshaped_on_x_lstm_1_out'] = rsh_on_x_lstm_1_out

        if not shared_p_v:
            # Aac policy output and action-sampling function:
            [self.on_logits, _, self.on_sample] = dense_aac_network(
                rsh_on_x_lstm_1_out,
                ac_space_depth=self.ac_space.one_hot_depth,
                linear_layer_ref=linear_layer_ref,
                name='aac_dense_pi'
            )

        # Second LSTM layer takes concatenated encoded 'external' state, LSTM_1 output,
        # last_action and `internal_state` (if present) tensors:
        on_stage2_2_input += [on_x_lstm_1_out]

        # Try: feed context instead of output
        #on_stage2_2_input = [reshape_lstm_1_state_out] + on_stage2_1_input

        # LSTM_2 full input:
        on_aac_x = tf.concat(on_stage2_2_input, axis=-1)

        self.debug['on_stage2_2_input'] = on_aac_x

        [on_x_lstm_2_out, self.on_lstm_2_init_state, self.on_lstm_2_state_out, self.on_lstm_2_state_pl_flatten] = \
            lstm_network(
                x=on_aac_x,
                lstm_sequence_length=self.on_time_length,
                lstm_class=lstm_class_ref,
                lstm_layers=(lstm_layers[-1],),
                static=static_rnn,
                name='lstm_2',
                **kwargs,
            )

        self.debug['on_x_lstm_2_out'] = on_x_lstm_2_out
        self.debug['self.on_lstm_2_state_out'] = self.on_lstm_2_state_out
        self.debug['self.on_lstm_2_state_pl_flatten'] = self.on_lstm_2_state_pl_flatten

        # Reshape back to [batch, flattened_depth], where batch = rnn_batch_dim * rnn_time_dim:
        x_shape_static = on_x_lstm_2_out.get_shape().as_list()
        rsh_on_x_lstm_2_out = tf.reshape(on_x_lstm_2_out, [x_shape_dynamic[0], x_shape_static[-1]])

        self.debug['reshaped_on_x_lstm_2_out'] = rsh_on_x_lstm_2_out

        if shared_p_v:
            [self.on_logits, self.on_vf, self.on_sample] = dense_aac_network(
                rsh_on_x_lstm_2_out,
                ac_space_depth=self.ac_space.one_hot_depth,
                linear_layer_ref=linear_layer_ref,
                name='aac_dense_pi_vfn'
            )

        else:
            # Aac value function:
            [_, self.on_vf, _] = dense_aac_network(
                rsh_on_x_lstm_2_out,
                ac_space_depth=self.ac_space.one_hot_depth,
                linear_layer_ref=linear_layer_ref,
                name='aac_dense_vfn'
            )

        # Concatenate LSTM placeholders, init. states and context:
        self.on_lstm_init_state = (self.on_lstm_1_init_state, self.on_lstm_2_init_state)
        self.on_lstm_state_out = (self.on_lstm_1_state_out, self.on_lstm_2_state_out)
        self.on_lstm_state_pl_flatten = self.on_lstm_1_state_pl_flatten + self.on_lstm_2_state_pl_flatten

        self.off_aac_x_encoded = {}
        for key in self.modes_to_encode:
            if isinstance(self.off_state_in[key], dict):  # got dictionary of data streams
                if self.share_encoder_params:
                    layer_name_template = 'encoded_{}_shared'
                else:
                    layer_name_template = 'encoded_{}_{}'
                encoded_streams = {
                    name: tf.compat.v1.layers.flatten(
                        self.state_encoder_class_ref(
                            x=stream,
                            ob_space=self.ob_space._shape[key][name],
                            ac_space=self.ac_space,
                            name=layer_name_template.format(key, name),
                            reuse=True,  # shared params for all streams in mode
                            **kwargs
                        )
                    )
                    for name, stream in self.off_state_in[key].items()
                }
                encoded_mode = tf.concat(
                    list(encoded_streams.values()),
                    axis=-1,
                    name='multi_encoded_{}'.format(key)
                )
            else:
                # Got single data stream:
                encoded_mode = tf.compat.v1.layers.flatten(
                    self.state_encoder_class_ref(
                        x=self.off_state_in[key],
                        ob_space=self.ob_space._shape[key],
                        ac_space=self.ac_space,
                        name='encoded_{}'.format(key),
                        reuse=True,
                        **kwargs
                    )
                )
            self.off_aac_x_encoded[key] = encoded_mode

        # off_aac_x = tf.concat(list(self.off_aac_x_encoded.values()), axis=-1, name='off_state_external_encoded')

        off_aac_x = self.off_aac_x_encoded['external']

        # Reshape rnn inputs for  batch training as [rnn_batch_dim, rnn_time_dim, flattened_depth]:
        x_shape_dynamic = tf.shape(input=off_aac_x)
        max_seq_len = tf.cast(x_shape_dynamic[0] / self.off_batch_size, tf.int32)
        x_shape_static = off_aac_x.get_shape().as_list()

        off_last_action_in = tf.reshape(
            self.off_last_a_in,
            [self.off_batch_size, max_seq_len, self.ac_space.encoded_depth]
        )
        off_last_r_in = tf.reshape(self.off_last_reward_in, [self.off_batch_size, max_seq_len, 1])

        off_aac_x = tf.reshape( off_aac_x, [self.off_batch_size, max_seq_len, np.prod(x_shape_static[1:])])

        # # Prepare `internal` state, if any:
        # if 'internal' in list(self.off_state_in.keys()):
        #     if self.encode_internal_state:
        #         # Use convolution encoder:
        #         off_x_internal = self.state_encoder_class_ref(
        #             x=self.off_state_in['internal'],
        #             ob_space=self.ob_space._shape['internal'],
        #             ac_space=self.ac_space,
        #             name='encoded_internal',
        #             reuse=True,
        #             **kwargs
        #         )
        #         x_int_shape_static = off_x_internal.get_shape().as_list()
        #         off_x_internal = [
        #             tf.reshape(off_x_internal, [self.off_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])])
        #         ]
        #     else:
        #         x_int_shape_static = self.off_state_in['internal'].get_shape().as_list()
        #         off_x_internal = tf.reshape(
        #             self.off_state_in['internal'],
        #             [self.off_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])]
        #         )
        #         off_x_internal = [off_x_internal]
        #
        # else:
        #     off_x_internal = []

        off_x_internal = self.off_aac_x_encoded['internal']

        x_int_shape_static = off_x_internal.get_shape().as_list()

        # Properly feed LSTM2:
        off_x_internal = tf.reshape(
            off_x_internal,
            [self.off_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])]
        )
        off_x_internal = [off_x_internal]

        if 'datetime' in list(self.off_state_in.keys()):
            x_dt_shape_static = self.off_state_in['datetime'].get_shape().as_list()
            off_x_dt = tf.reshape(
                self.off_state_in['datetime'],
                [self.off_batch_size, max_seq_len, np.prod(x_dt_shape_static[1:])]
            )
            off_x_dt = [off_x_dt]

        else:
            off_x_dt = []

        # off_stage2_1_input = [off_aac_x,  off_last_action_in, off_last_reward_in] + off_x_dt
        off_stage2_1_input = [off_aac_x, off_last_r_in]  # + off_x_dt

        # off_stage2_2_input = [off_aac_x,  off_last_action_in, off_last_reward_in] + off_x_internal + off_x_dt
        off_stage2_2_input = [off_aac_x,  off_last_action_in] + off_x_internal  # + off_x_dt

        off_aac_x = tf.concat(off_stage2_1_input, axis=-1)

        [off_x_lstm_1_out, _, _, self.off_lstm_1_state_pl_flatten] =\
            lstm_network(
                off_aac_x,
                self.off_time_length,
                lstm_class_ref,
                (lstm_layers[0],),
                name='lstm_1',
                static=static_rnn,
                reuse=True,
                **kwargs,
            )

        # Reshape back to [batch, flattened_depth], where batch = rnn_batch_dim * rnn_time_dim:
        x_shape_static = off_x_lstm_1_out.get_shape().as_list()
        rsh_off_x_lstm_1_out = tf.reshape(off_x_lstm_1_out, [x_shape_dynamic[0], x_shape_static[-1]])

        if not shared_p_v:
            [self.off_logits, _, _] =\
                dense_aac_network(
                    rsh_off_x_lstm_1_out,
                    ac_space_depth=self.ac_space.one_hot_depth,
                    linear_layer_ref=linear_layer_ref,
                    name='aac_dense_pi',
                    reuse=True
                )

        off_stage2_2_input += [off_x_lstm_1_out]

        # LSTM_2 full input:
        off_aac_x = tf.concat(off_stage2_2_input, axis=-1)

        [off_x_lstm_2_out, _, _, self.off_lstm_2_state_pl_flatten] = \
            lstm_network(
                off_aac_x,
                self.off_time_length,
                lstm_class_ref,
                (lstm_layers[-1],),
                name='lstm_2',
                static=static_rnn,
                reuse=True,
                **kwargs,
            )

        # Reshape back to [batch, flattened_depth], where batch = rnn_batch_dim * rnn_time_dim:
        x_shape_static = off_x_lstm_2_out.get_shape().as_list()
        rsh_off_x_lstm_2_out = tf.reshape(off_x_lstm_2_out, [x_shape_dynamic[0], x_shape_static[-1]])

        if shared_p_v:
            [self.off_logits, self.off_vf, _] = dense_aac_network(
                rsh_off_x_lstm_2_out,
                ac_space_depth=self.ac_space.one_hot_depth,
                linear_layer_ref=linear_layer_ref,
                name='aac_dense_pi_vfn',
                reuse=True
            )
        else:
            # Aac value function:
            [_, self.off_vf, _] = dense_aac_network(
                rsh_off_x_lstm_2_out,
                ac_space_depth=self.ac_space.one_hot_depth,
                linear_layer_ref=linear_layer_ref,
                name='aac_dense_vfn',
                reuse=True
            )

        # Concatenate LSTM states:
        self.off_lstm_state_pl_flatten = self.off_lstm_1_state_pl_flatten + self.off_lstm_2_state_pl_flatten

        if False:  # TEMP DISABLE
            # Aux1:
            # `Pixel control` network.
            #
            # Define pixels-change estimation function:
            # Yes, it rather env-specific but for atari case it is handy to do it here, see self.get_pc_target():
            [self.pc_change_state_in, self.pc_change_last_state_in, self.pc_target] =\
                pixel_change_2d_estimator(ob_space['external'], **kwargs)

            self.pc_batch_size = self.off_batch_size
            self.pc_time_length = self.off_time_length

            self.pc_state_in = self.off_state_in
            self.pc_a_r_in = self.off_a_r_in
            self.pc_lstm_state_pl_flatten = self.off_lstm_state_pl_flatten

            # Shared conv and lstm nets, same off-policy batch:
            pc_x = rsh_off_x_lstm_2_out

            # PC duelling Q-network, outputs [None, 20, 20, ac_size] Q-features tensor:
            self.pc_q = duelling_pc_network(pc_x, self.ac_space, linear_layer_ref=linear_layer_ref, **kwargs)

        # Aux2:
        # `Value function replay` network.
        #
        # VR network is fully shared with ppo network but with `value` only output:
        # and has same off-policy batch pass with off_ppo network:
        self.vr_batch_size = self.off_batch_size
        self.vr_time_length = self.off_time_length

        self.vr_state_in = self.off_state_in
        self.vr_last_a_in = self.off_last_a_in
        self.vr_last_reward_in = self.off_last_reward_in

        self.vr_lstm_state_pl_flatten = self.off_lstm_state_pl_flatten
        self.vr_value = self.off_vf

        # Aux3:
        # `Reward prediction` network.
        self.rp_batch_size = tf.compat.v1.placeholder(tf.int32, name='rp_batch_size')

        # Shared encoded output:
        rp_x = {}
        for key in self.rp_state_in.keys():
            if 'external' in key:
                if isinstance(self.rp_state_in[key], dict):  # got dictionary of data streams
                    if self.share_encoder_params:
                        layer_name_template = 'encoded_{}_shared'
                    else:
                        layer_name_template = 'encoded_{}_{}'
                    encoded_streams = {
                        name: tf.compat.v1.layers.flatten(
                            self.state_encoder_class_ref(
                                x=stream,
                                ob_space=self.ob_space._shape[key][name],
                                ac_space=self.ac_space,
                                name=layer_name_template.format(key, name),
                                reuse=True,  # shared params for all streams in mode
                                **kwargs
                            )
                        )
                        for name, stream in self.rp_state_in[key].items()
                    }
                    encoded_mode = tf.concat(
                        list(encoded_streams.values()),
                        axis=-1,
                        name='multi_encoded_{}'.format(key)
                    )
                else:
                    # Got single data stream:
                    encoded_mode = tf.compat.v1.layers.flatten(
                        self.state_encoder_class_ref(
                            x=self.rp_state_in[key],
                            ob_space=self.ob_space._shape,
                            ac_space=self.ac_space,
                            name='encoded_{}'.format(key),
                            reuse=True,
                            **kwargs
                        )
                    )
                rp_x[key] = encoded_mode

        rp_x = tf.concat(list(rp_x.values()), axis=-1, name='rp_state_external_encoded')

        # Flatten batch-wise:
        rp_x_shape_static = rp_x.get_shape().as_list()
        rp_x = tf.reshape(rp_x, [self.rp_batch_size, np.prod(rp_x_shape_static[1:]) * (self.rp_sequence_size-1)])

        # RP output:
        self.rp_logits = dense_rp_network(rp_x, linear_layer_ref=linear_layer_ref)

        # Batch-norm related:
        self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        # Add moving averages to save list:
        moving_var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, tf.compat.v1.get_variable_scope().name + '.*moving.*')
        renorm_var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, tf.compat.v1.get_variable_scope().name + '.*renorm.*')

        # What to save:
        self.var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, tf.compat.v1.get_variable_scope().name)
        self.var_list += moving_var_list + renorm_var_list

        # Callbacks:
        if self.aux_estimate:
            pass
            # TEMP DISABLE: due to computation costs
            # do not use pixel change aux. task; otherwise enable lines 533 - 553 & 640:
            if False:
                self.callback['pixel_change'] = self.get_pc_target

        # print('policy_debug_dict:\n', self.debug)


class AacStackedRL2Policy(StackedLstmPolicy):
    """
    Attempt to implement two-level RL^2
    This policy class in conjunction with DataDomain classes from btgym_tf2.datafeed
    is aimed to implement RL^2 algorithm by Duan et al.

    Paper:
    `FAST REINFORCEMENT LEARNING VIA SLOW REINFORCEMENT LEARNING`,
    https://arxiv.org/pdf/1611.02779.pdf

    The only difference from Base policy is `get_initial_features()` method, which has been changed
    either to reset RNN context to zero-state or return context from the end of previous episode,
    depending on episode metadata received or `lstm_2_init_period' parameter.
    """
    def __init__(self, lstm_2_init_period=50, **kwargs):
        super(AacStackedRL2Policy, self).__init__(**kwargs)
        self.current_trial_num = -1  # always give initial context at first call
        self.lstm_2_init_period = lstm_2_init_period
        self.current_ep_num = 0

    def get_initial_features(self, state, context=None):
        """
        Returns RNN initial context.
        RNN_1 (lower) context is reset at every call.

        RNN_2 (upper) context is reset:
            - every `lstm_2_init_period' episodes;
            - episode  initial `state` `trial_num` metadata has been changed form last call (new train trial started);
            - episode metatdata `type` is non-zero (test episode);
            - no context arg is provided (initial episode of training);
            - ... else carries context on to new episode;

        Episode metadata are provided by DataTrialIterator, which is shaping Trial data distribution in this case,
        and delivered through env.strategy as separate key in observation dictionary.

        Args:
            state:      initial episode state (result of env.reset())
            context:    last previous episode RNN state (last_context of runner)

        Returns:
            2_RNN zero-state tuple.

        Raises:
            KeyError if [`metadata`]:[`trial_num`,`type`] keys not found
        """
        try:
            sess = tf.compat.v1.get_default_session()
            new_context = list(sess.run(self.on_lstm_init_state))
            if state['metadata']['trial_num'] != self.current_trial_num\
                    or context is None\
                    or state['metadata']['type']\
                    or self.current_ep_num % self.lstm_2_init_period == 0:
                # Assume new/initial trial or test sample, reset_1, 2 context:
                pass #print('RL^2 policy context 1, 2 reset')

            else:
                # Asssume same training trial, keep context_2 same as received:
                new_context[-1] = context[-1]
                #print('RL^2 policy context 1, reset')
            # Back to tuple:
            new_context = tuple(new_context)
            # Keep trial number:
            self.current_trial_num = state['metadata']['trial_num']

        except KeyError:
            raise KeyError(
                'RL^2 policy: expected observation state dict. to have keys [`metadata`]:[`trial_num`,`type`]; got: {}'.
                format(state.keys())
            )
        self.current_ep_num +=1
        return new_context

