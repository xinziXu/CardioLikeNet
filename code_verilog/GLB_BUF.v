`timescale  1ns/100ps
module GLB_BUF #(parameter  INPUT_DW = 12,
    DATA_OUT_DW = 8,
    FEATURE_SUM_DW = INPUT_DW + 4,
    ENCODER_LENGTH_IN = 256,
    INTEVAL_DW= $clog2(ENCODER_LENGTH_IN+1),
    SAVE_NUM_BEATS = 9,
    NUM_FEAS_MI = 9,
    NUM_LEADS = 12,
    DATA_BQ_DW = 32,
    PE_NUM = 32,
    SPAD_DEPTH = 8,
    // CNN21_OUT = 2048,
    
    ENCODER_PADDING_PRE = 3,
    ENCODER_PADDING_POST = 1,
    ENCODER_STRIDE = 4,
    ENCODER_KS = 8,
    ENCODER_CHIN = 1,
    ENCODER_CHOUT = 32,
    ENCODER_LENGTH_OUT = 64,
    LSTM_NUM_LAYERS = 2,
    LSTM_NUM_DIR = 2,
    LSTM_HS = 32,
    DCNN1_CHIN =LSTM_NUM_DIR * LSTM_HS ,
    DCNN1_LENGTH_IN = ENCODER_LENGTH_OUT,
    DCNN1_CHOUT = 32,
    DCNN_STRIDE = 2,
    DCNN_PADDING = 3,
    DCNN_KS = 8,
    DCNN1_LENGTH_OUT =  DCNN_STRIDE * (LSTM_NUM_DIR*LSTM_HS - 1) -  2 * DCNN_PADDING + DCNN_KS,
    CNN11_LENGTH_IN = DCNN1_LENGTH_OUT,
    CNN11_CHIN = DCNN1_CHOUT,
    CNN11_CHOUT = 32,
    CNN_PADDING = 2,
    CNN_KS = 5,
    CNN11_LENGTH_OUT = DCNN1_LENGTH_OUT,
    CNN12_LENGTH_IN = CNN11_LENGTH_OUT,
    CNN12_CHIN = CNN11_CHOUT,
    CNN12_CHOUT = 16,
    CNN12_LENGTH_OUT =  DCNN1_LENGTH_OUT,
    DCNN2_LENGTH_IN = CNN12_LENGTH_OUT,
    DCNN2_CHIN = CNN12_CHOUT,
    DCNN2_CHOUT =  8,
    DCNN2_LENGTH_OUT = DCNN_STRIDE * (DCNN2_LENGTH_IN - 1) -  2 * DCNN_PADDING + DCNN_KS,
    CNN21_LENGTH_IN = DCNN2_LENGTH_OUT,
    CNN21_CHIN = DCNN2_CHOUT,
    CNN21_CHOUT = 8,
    CNN21_LENGTH_OUT = DCNN2_LENGTH_OUT,
    CNN22_LENGTH_IN = DCNN2_LENGTH_OUT,
    CNN22_CHIN = CNN21_CHOUT,
    CNN22_CHOUT = 4,
    CNN22_LENGTH_OUT =  DCNN2_LENGTH_OUT,
    ACTIVATION_BUF_LEN1 = LSTM_HS * ENCODER_LENGTH_OUT, 
    ACTIVATION_BUF_LEN2 = ENCODER_CHOUT * ENCODER_LENGTH_OUT,
    ACTIVATION_BUF_LEN3 = LSTM_HS * ENCODER_LENGTH_OUT,
    ACTIVATION_BUF_LEN4 = DCNN1_LENGTH_OUT * DCNN1_CHOUT,
    NUM_WAVE = 4,
    DIR_DW = 2,
    LABEL_DW = 2,
    EMB_DW = 2,
    QRS_EMB_LEN = 24,
    T_EMB_LEN = 25,
    ANN_WB_DW = 16,
    FEATURE_DIM = 22,
    FEATURE_DIM_MI = 64,
    ANN_HIDDEN_DIM =32)
    (input wclk,
    input rst_n,
    input [INPUT_DW*ENCODER_LENGTH_IN-1:0] input_signal,
    // input [LABEL_DW*ENCODER_LENGTH_IN-1:0] softmax_out_all,
    // input [DATA_OUT_DW*CNN21_OUT-1:0] cnn21_out,
    input [3:0] top_state,
    input [3:0] seg_state,
    input [2:0] lstm_top_state, 
    input [2:0] decoder_top_state,
    input [4:0] ann_state,
    input cnn22_is_first_2d,  
    input [$clog2(PE_NUM+1)-1:0] cnt_cho_32,
    input [PE_NUM*DATA_BQ_DW-1: 0] pe_out_32b_all,
    input signed [DATA_OUT_DW-1: 0] encoder_out, 
    input encoder_out_vld, //encoder
    input signed [2*DATA_OUT_DW-1: 0] lstm_hidden_cat, //lstm out
    input signed [DATA_OUT_DW-1: 0] decoder_out, // dcnn1, cnn11, cnn12
    input [2*DATA_OUT_DW-1: 0] decoder_out_cat, //dcnn2, cnn21, cnn22
    input [LABEL_DW-1:0] softmax_out,
    input lstm_hidden_unit_vld,
    input decoder_out_vld,
    input dcnn1_temp_value_vld,
    input dcnn1_transfer_temp_value_en,
    input dcnn1_temp_rst,
    input [1:0] encoder_shift_en,
    input lstm_xt_shift_en,
    input [2*PE_NUM-1:0] shift_crl_all,
    input [PE_NUM-1:0] cnt_bt_all,
    input [$clog2(NUM_LEADS+1)-1 : 0] cnt_lead,
    output reg [DATA_OUT_DW*ACTIVATION_BUF_LEN1-1:0] act_sr1,
    output reg [DATA_OUT_DW*ACTIVATION_BUF_LEN2-1:0] act_sr2,
    output reg [DATA_OUT_DW*ACTIVATION_BUF_LEN3-1:0] act_sr3,
    output reg [DATA_OUT_DW*ACTIVATION_BUF_LEN4-1:0] act_sr4,
    output reg [5*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1:0] feature_rb,
    
    input [3:0] post_state,
    input [4:0] refine_state,
    input [$clog2(ENCODER_LENGTH_IN+1)-1:0] wave_duration,
    input modify_en,
    input connection_shift,
    input refine_shift_re,
    input refine_shift,
    input emb_shift,
    input feature_shift,
    input [1:0] save_fea_en,
    input [INTEVAL_DW-1:0] rr_pre_d,
    input signed [INPUT_DW-1:0] r_amp_d,
    input signed [INPUT_DW-1:0] t_amp_d,
    input signed [INPUT_DW-1:0] p_amp_d,
    input signed [INPUT_DW-1:0] q_amp_d,
    input [INTEVAL_DW-1:0] qrs_d,
    input signed [INPUT_DW-1:0] s_amp_d,
    input [INTEVAL_DW-1:0] rr_pre,
    input signed [INPUT_DW-1:0] r_amp,
    input signed [INPUT_DW-1:0] t_amp,
    input signed [INPUT_DW-1:0] p_amp,
    input signed [INPUT_DW-1:0] q_amp,
    input [INTEVAL_DW-1:0] qrs,
    input signed [INPUT_DW-1:0] s_amp,
    // input signed [INPUT_DW - 1: 0] r_amp_r_amp_ave,
    // input signed [INPUT_DW - 1: 0] q_amp_q_amp_ave,
    // input signed [INPUT_DW - 1: 0] s_amp_s_amp_ave,
    // input signed [INPUT_DW - 1: 0] p_amp_p_amp_ave,
    // input signed [INPUT_DW - 1: 0] t_amp_t_amp_ave,

    // input signed [INPUT_DW - 1: 0] r_amp_t_amp,
    input signed [INPUT_DW - 1: 0] q_amp_iso,
    input signed [INPUT_DW - 1: 0] s_amp_iso,
    input signed [INPUT_DW - 1: 0] t_amp_iso,
    input signed [INPUT_DW - 1: 0] r_amp_iso,
    input signed [INPUT_DW - 1: 0] st_amp_iso,
    input signed [INPUT_DW - 1: 0] st_slo,
    input signed [FEATURE_SUM_DW - 1: 0] r_amp_t_amp_sum,
    input signed [FEATURE_SUM_DW - 1: 0] q_amp_iso_sum,
    input signed [FEATURE_SUM_DW - 1: 0] s_amp_iso_sum,
    input signed [FEATURE_SUM_DW - 1: 0] t_amp_iso_sum,
    input signed [FEATURE_SUM_DW - 1: 0] r_amp_iso_sum,
    input signed [FEATURE_SUM_DW - 1: 0] st_amp_iso_sum,
    input signed [FEATURE_SUM_DW - 1: 0] st_slo_sum,
    input  [EMB_DW*QRS_EMB_LEN-1:0] qrs_emb_buffer,
    input  [EMB_DW*T_EMB_LEN-1:0] t_emb_buffer,
    input signed [DIR_DW-1:0] t_dir,
  

    
    input [1:0] ann_shift,
    
    input input_init_en,
    input ann_mi_1,
    input ann_mi_2,
    input [ANN_WB_DW + INPUT_DW -1:0] ann_out,
    input [ANN_WB_DW + FEATURE_SUM_DW -1:0] ann_out_mi,
    input  [FEATURE_DIM*INPUT_DW-1:0] feature_matrix,
    input [NUM_FEAS_MI * FEATURE_SUM_DW*2 + EMB_DW * QRS_EMB_LEN + EMB_DW*T_EMB_LEN  + DIR_DW-1:0] feature_matrix_mi,
    input ann_out_vld,
    input ann_hidden_out_vld);
    
    localparam dcnn1 =  3'b001; //from sram
    localparam cnn11 = 3'b011;
    localparam cnn12 = 3'b111;
    localparam dcnn2 = 3'b101;
    localparam cnn21  = 3'b110 ;
    localparam cnn22 = 3'b010;

    localparam idle    = 4'b0000;
    localparam encoder = 4'b0001;
    localparam lstm    = 4'b0010;
    localparam decoder = 4'b0100;
    localparam seg_done    = 4'b1000;    

    localparam seg_network = 4'b0010;  //seg-network
    localparam post = 4'b0011;
    localparam feature_map = 4'b0111;
    localparam ann  = 4'b0101;
    localparam top_done        = 4'b1111;

    localparam connection = 4'b0001;
    localparam prepare_glb = 4'b0101;
    localparam refine = 4'b0011; 
    localparam embedding = 4'b0111;
    localparam post_done    = 4'b1000;
    localparam refine_idle    = 5'd0;
    localparam prepare = 5'd1; //qrs_info, p_info, t_info
    localparam select_rough_qs = 5'd2; //
    localparam determine_r = 5'd3;
    localparam select_rough_t = 5'd4; 
    localparam determine_t =  5'd5;
    localparam determine_t_off =  5'd6;
    localparam determine_t_wait =  5'd18;
    localparam determine_t_on = 5'd7;
    localparam select_rough_p = 5'd8; 
    localparam determine_p = 5'd9;
    localparam determine_p_wait = 5'd19;
    localparam determine_p_off = 5'd10;
    localparam determine_p_on = 5'd11 ;
    localparam determine_q =  5'd12;
    localparam determine_q_re =  5'd13;
    localparam determine_s =  5'd14; 
    localparam determine_s_re =  5'd15;
    localparam s_st_modify =  5'd16;
    localparam refine_finish    =  5'd17;
    localparam mi_points = 5'd18;

    localparam ENCODER_INPUT_PAD_LEN = ENCODER_LENGTH_IN + ENCODER_PADDING_PRE+ ENCODER_PADDING_POST;
    reg act_sr_init_done;
    integer shift_id_cnn12_1;
    integer  shift_id_cnn21;

    // wire [FEATURE_SUM_DW-1:0] act_sr4_test;
    // assign act_sr4_test = act_sr4[FEATURE_SUM_DW-1:0];
//    wire signed [FEATURE_SUM_DW -1:0] act_sr4_mem[FEATURE_DIM_MI-1:0];

       
//    genvar   test_id;
//    generate
//    for (test_id = 0; test_id < FEATURE_DIM_MI; test_id = test_id + 1) begin: test
//        assign act_sr4_mem[test_id] = act_sr4[test_id*(FEATURE_SUM_DW)+(FEATURE_SUM_DW)-1- :(FEATURE_SUM_DW)];
//    end        
//    endgenerate

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            act_sr1           <= 0; //change
            // act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= softmax_out_all;
            act_sr_init_done <= 0;
        end
        else begin
            if (top_state == seg_network) begin
                if (seg_state == encoder) begin
                    if (!act_sr_init_done) begin
                        act_sr1[ENCODER_INPUT_PAD_LEN*INPUT_DW-1:0]   <= {{ENCODER_PADDING_POST* INPUT_DW {1'b0}},input_signal,{ENCODER_PADDING_PRE* INPUT_DW {1'b0}}}; // add padding
                        act_sr_init_done <= 1;
                    end
                    else begin
                        act_sr_init_done <= 1;
                        if (encoder_shift_en) 
                            act_sr1[ENCODER_INPUT_PAD_LEN*INPUT_DW-1:0] <= {act_sr1[ENCODER_STRIDE*INPUT_DW-1:0], act_sr1[ENCODER_INPUT_PAD_LEN* INPUT_DW -1:ENCODER_STRIDE* INPUT_DW]}; // shift for input
                        else
                            act_sr1 <= act_sr1 ;
                    end 
                end
                if (seg_state == lstm) begin
                    if (lstm_top_state == 3'd1) begin
                        if (lstm_hidden_unit_vld) //ht, T64, T63, ..., T1
                            act_sr1 <= { lstm_hidden_cat, act_sr1[ACTIVATION_BUF_LEN1*DATA_OUT_DW-1:2*DATA_OUT_DW] };
                        else
                            act_sr1 <=  act_sr1;                    
                    end
                    else if (lstm_top_state == 3'd2) begin
                        act_sr1 <=  act_sr1;
                    end
                    else  if (lstm_top_state == 3'd3) begin //act_sr1, act_sr2
                        if (lstm_xt_shift_en) //ht ->
                            act_sr1 <= {act_sr1[DATA_OUT_DW*LSTM_HS-1:0], act_sr1[ACTIVATION_BUF_LEN1*DATA_OUT_DW-1:DATA_OUT_DW*LSTM_HS]};
                        else
                            act_sr1 <=  act_sr1;
                    end
                    else  if (lstm_top_state == 3'd4) begin //act_sr1, act_sr2
                        if (lstm_xt_shift_en) //ht <-
                            act_sr1 <= {act_sr1[(ACTIVATION_BUF_LEN1-LSTM_HS)*DATA_OUT_DW-1:0],act_sr1[DATA_OUT_DW*ACTIVATION_BUF_LEN1-1-:DATA_OUT_DW*LSTM_HS]};
                        else
                            act_sr1 <=  act_sr1;
                    end
                    else if (lstm_top_state == 3'd5) begin
                        act_sr1 <= 0; //reset
                    end
                end
                else if (seg_state == decoder) begin
                    if (decoder_top_state == dcnn1) begin
                        if (dcnn1_temp_value_vld) begin
                            act_sr1 [(DCNN1_LENGTH_OUT-2)*DATA_BQ_DW-1:0] <= {pe_out_32b_all[DATA_BQ_DW-1-:DATA_BQ_DW],act_sr1[(DCNN1_LENGTH_OUT-2)*DATA_BQ_DW-1:DATA_BQ_DW]};
                        end
                        else begin
                            if (dcnn1_temp_rst) begin
                                act_sr1 <= 0;
                            end
                            else begin
                                if (dcnn1_transfer_temp_value_en) act_sr1 [(DCNN1_LENGTH_OUT-2)*DATA_BQ_DW-1:0] <= {act_sr1[DATA_BQ_DW-1:0],act_sr1[(DCNN1_LENGTH_OUT-2)*DATA_BQ_DW-1:DATA_BQ_DW]};
                                else act_sr1 <= act_sr1;
                            end
                        end
                    end
                    else if (decoder_top_state == cnn11) begin
                        if ((decoder_out_vld) & (cnt_cho_32 <CNN11_CHOUT/2)) begin
                            act_sr1 <= {decoder_out, act_sr1[CNN11_LENGTH_OUT/2*CNN11_CHOUT*DATA_OUT_DW-1:DATA_OUT_DW]}; 
                        end
                        else act_sr1<=act_sr1;
                    end
                    else if (decoder_top_state == cnn12) begin
                        for (shift_id_cnn12_1=0;shift_id_cnn12_1<CNN12_CHIN/2;shift_id_cnn12_1=shift_id_cnn12_1+1) begin
                            if  (shift_crl_all[2*(shift_id_cnn12_1)+2-1-:2] == 1)
                                act_sr1[(shift_id_cnn12_1+1)*CNN12_LENGTH_IN*DATA_OUT_DW-1-:CNN12_LENGTH_IN*DATA_OUT_DW] <= {act_sr1[shift_id_cnn12_1*CNN12_LENGTH_IN*DATA_OUT_DW+DATA_OUT_DW-1-:DATA_OUT_DW],act_sr1[(shift_id_cnn12_1+1)*CNN12_LENGTH_IN*DATA_OUT_DW-1-:(CNN12_LENGTH_IN-1)*DATA_OUT_DW] };
                            else if (shift_crl_all[2*(shift_id_cnn12_1)+2-1-:2] == 2)
                                act_sr1[(shift_id_cnn12_1+1)*CNN12_LENGTH_IN*DATA_OUT_DW-1-:CNN12_LENGTH_IN*DATA_OUT_DW] <= {act_sr1[shift_id_cnn12_1*CNN12_LENGTH_IN*DATA_OUT_DW+4*DATA_OUT_DW-1-:4*DATA_OUT_DW],act_sr1[(shift_id_cnn12_1+1)*CNN12_LENGTH_IN*DATA_OUT_DW-1-:(CNN12_LENGTH_IN-4)*DATA_OUT_DW] };
                            else act_sr1[(shift_id_cnn12_1+1)*CNN12_LENGTH_IN*DATA_OUT_DW-1-:CNN12_LENGTH_IN*DATA_OUT_DW] <= act_sr1 [(shift_id_cnn12_1+1)*CNN12_LENGTH_IN*DATA_OUT_DW-1-:CNN12_LENGTH_IN*DATA_OUT_DW] ;
                        end
                    end
                    else if (decoder_top_state == dcnn2) begin
                        if (decoder_out_vld)  begin
                            act_sr1[DCNN2_CHOUT/2*DCNN2_LENGTH_OUT*DATA_OUT_DW-1:0] <= {decoder_out_cat[DATA_OUT_DW-1:0], act_sr1[DCNN2_CHOUT/2*DCNN2_LENGTH_OUT*DATA_OUT_DW-1:DATA_OUT_DW]};
                            act_sr1[DCNN2_CHOUT*DCNN2_LENGTH_OUT*DATA_OUT_DW-1:DCNN2_CHOUT/2*DCNN2_LENGTH_OUT*DATA_OUT_DW] <= {decoder_out_cat[2*DATA_OUT_DW-1:DATA_OUT_DW], act_sr1[DCNN2_CHOUT*DCNN2_LENGTH_OUT*DATA_OUT_DW-1:DCNN2_CHOUT/2*DCNN2_LENGTH_OUT*DATA_OUT_DW+DATA_OUT_DW]};
                        end
                        else act_sr1<= act_sr1;
                    end
                    else if (decoder_top_state == cnn21) begin
                        for (shift_id_cnn21=0;shift_id_cnn21<CNN21_CHIN;shift_id_cnn21=shift_id_cnn21+1) begin
                            if (shift_crl_all[2*(shift_id_cnn21)+2-1-:2] == 1) begin
                                act_sr1[(shift_id_cnn21+1)*CNN21_LENGTH_IN*DATA_OUT_DW-1-:CNN21_LENGTH_IN*DATA_OUT_DW] <= {act_sr1[shift_id_cnn21*CNN21_LENGTH_IN*DATA_OUT_DW+DATA_OUT_DW-1-:DATA_OUT_DW],act_sr1[(shift_id_cnn21+1)*CNN21_LENGTH_IN*DATA_OUT_DW-1-:(CNN21_LENGTH_IN-1)*DATA_OUT_DW] };
                            end
                            else if (shift_crl_all[2*(shift_id_cnn21)+2-1-:2] == 2) begin
                                act_sr1[(shift_id_cnn21+1)*CNN21_LENGTH_IN*DATA_OUT_DW-1-:CNN21_LENGTH_IN*DATA_OUT_DW] <= {act_sr1[shift_id_cnn21*CNN21_LENGTH_IN*DATA_OUT_DW+4*DATA_OUT_DW-1-:4*DATA_OUT_DW],act_sr1[(shift_id_cnn21+1)*CNN21_LENGTH_IN*DATA_OUT_DW-1-:(CNN21_LENGTH_IN-4)*DATA_OUT_DW] };
                            end
                            else begin
                                act_sr1[(shift_id_cnn21+1)*CNN21_LENGTH_IN*DATA_OUT_DW-1-:CNN21_LENGTH_IN*DATA_OUT_DW]<=act_sr1[(shift_id_cnn21+1)*CNN21_LENGTH_IN*DATA_OUT_DW-1-:CNN21_LENGTH_IN*DATA_OUT_DW];
                            end
                        end
                    end
                    else if (decoder_top_state == cnn22)   begin
                        if(decoder_out_vld) begin
                            if (!cnn22_is_first_2d) begin
                                act_sr1[LABEL_DW*CNN22_LENGTH_OUT-1:0] <= {softmax_out, act_sr1[LABEL_DW*CNN22_LENGTH_OUT-1:LABEL_DW]};                        
                                end
                            else act_sr1 <= act_sr1;
                            // if (cnn22_is_first_2d) begin
                            //     act_sr1[CNN22_CHOUT/4*CNN22_LENGTH_OUT*DATA_OUT_DW-1:0] <= {decoder_out_cat[DATA_OUT_DW-1:0],act_sr1 [CNN22_CHOUT/4*CNN22_LENGTH_OUT*DATA_OUT_DW-1:DATA_OUT_DW]};
                            //     act_sr1[CNN22_CHOUT/2*CNN22_LENGTH_OUT*DATA_OUT_DW-1:CNN22_CHOUT/4*CNN22_LENGTH_OUT*DATA_OUT_DW] <= {decoder_out_cat[2*DATA_OUT_DW-1:DATA_OUT_DW],act_sr1[CNN22_CHOUT/2*CNN22_LENGTH_OUT*DATA_OUT_DW-1:CNN22_CHOUT/4*CNN22_LENGTH_OUT*DATA_OUT_DW+DATA_OUT_DW]};
                            
                            // end
                            // else begin
                            //     act_sr1[CNN22_CHOUT*3/4*CNN22_LENGTH_OUT*DATA_OUT_DW-1:CNN22_CHOUT/2*CNN22_LENGTH_OUT*DATA_OUT_DW] <= {decoder_out_cat[DATA_OUT_DW-1:0],act_sr1[CNN22_CHOUT*3/4*CNN22_LENGTH_OUT*DATA_OUT_DW-1:CNN22_CHOUT/2*CNN22_LENGTH_OUT*DATA_OUT_DW+DATA_OUT_DW]};
                            //     act_sr1[CNN22_CHOUT*CNN22_LENGTH_OUT*DATA_OUT_DW-1:CNN22_CHOUT*3/4*CNN22_LENGTH_OUT*DATA_OUT_DW] <=  {decoder_out_cat[2*DATA_OUT_DW-1:DATA_OUT_DW],act_sr1[CNN22_CHOUT*CNN22_LENGTH_OUT*DATA_OUT_DW-1:CNN22_CHOUT*3/4*CNN22_LENGTH_OUT*DATA_OUT_DW+DATA_OUT_DW]};
                            // end
                        end
                        else act_sr1 <= act_sr1;
                    end
                    
                end
            end
            else if (top_state == post) begin
                // if (post_state == prepare_glb) begin
                //     act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= softmax_out_all;
                // end
                if (post_state == connection) begin
                    if (connection_shift) begin
                        if (modify_en) begin
                            case (wave_duration)
                                0: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {act_sr1[LABEL_DW-1:0],act_sr1[ENCODER_LENGTH_IN*LABEL_DW-1:LABEL_DW] };
                                1: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {{2{act_sr1[LABEL_DW-1:0]}},act_sr1[(ENCODER_LENGTH_IN-1)*LABEL_DW-1:LABEL_DW] };
                                2: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {{3{act_sr1[LABEL_DW-1:0]}},act_sr1[(ENCODER_LENGTH_IN-2)*LABEL_DW-1:LABEL_DW] };
                                3: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {{4{act_sr1[LABEL_DW-1:0]}},act_sr1[(ENCODER_LENGTH_IN-3)*LABEL_DW-1:LABEL_DW] };
                                4: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {{5{act_sr1[LABEL_DW-1:0]}},act_sr1[(ENCODER_LENGTH_IN-4)*LABEL_DW-1:LABEL_DW] };
                                5: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {{6{act_sr1[LABEL_DW-1:0]}},act_sr1[(ENCODER_LENGTH_IN-5)*LABEL_DW-1:LABEL_DW] };
                                6: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {{7{act_sr1[LABEL_DW-1:0]}},act_sr1[(ENCODER_LENGTH_IN-6)*LABEL_DW-1:LABEL_DW] };
                                7: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {{8{act_sr1[LABEL_DW-1:0]}},act_sr1[(ENCODER_LENGTH_IN-7)*LABEL_DW-1:LABEL_DW] };
                                8: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {{9{act_sr1[LABEL_DW-1:0]}},act_sr1[(ENCODER_LENGTH_IN-8)*LABEL_DW-1:LABEL_DW] };
                                9: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {{10{act_sr1[LABEL_DW-1:0]}},act_sr1[(ENCODER_LENGTH_IN-9)*LABEL_DW-1:LABEL_DW] };
                                10: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {{11{act_sr1[LABEL_DW-1:0]}},act_sr1[(ENCODER_LENGTH_IN-10)*LABEL_DW-1:LABEL_DW] };
                                11: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {{12{act_sr1[LABEL_DW-1:0]}},act_sr1[(ENCODER_LENGTH_IN-10)*LABEL_DW-1:LABEL_DW] };
                                12: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {{13{act_sr1[LABEL_DW-1:0]}},act_sr1[(ENCODER_LENGTH_IN-10)*LABEL_DW-1:LABEL_DW] };
                                13: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {{14{act_sr1[LABEL_DW-1:0]}},act_sr1[(ENCODER_LENGTH_IN-10)*LABEL_DW-1:LABEL_DW] };
                                14: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {{15{act_sr1[LABEL_DW-1:0]}},act_sr1[(ENCODER_LENGTH_IN-10)*LABEL_DW-1:LABEL_DW] };
                                15: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {{16{act_sr1[LABEL_DW-1:0]}},act_sr1[(ENCODER_LENGTH_IN-10)*LABEL_DW-1:LABEL_DW] };
                                
                                default: act_sr1[LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {act_sr1[LABEL_DW-1:0],act_sr1[ENCODER_LENGTH_IN*LABEL_DW-1:LABEL_DW] };
                            endcase
                        end
                        else begin
                            act_sr1 [LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {act_sr1[LABEL_DW-1:0],act_sr1[ENCODER_LENGTH_IN*LABEL_DW-1:LABEL_DW] };
                        end
                    end
                    else act_sr1<= act_sr1;
                end
                else if (post_state == refine) begin
                    if ((refine_shift) & (refine_state == prepare)) begin
                        act_sr1 [LABEL_DW*ENCODER_LENGTH_IN-1:0] <= {act_sr1[LABEL_DW-1:0],act_sr1[ENCODER_LENGTH_IN*LABEL_DW-1:LABEL_DW] };
                    end
                    else begin
                        act_sr1<= act_sr1;
                    end
                end
                

                // else if (post_state == post_done) act_sr1 <= 0;
                else ;
            end
            else if (top_state == ann) begin
                if ((ann_state == 5'd1) | (ann_state == 5'd2)) begin
                    if (ann_hidden_out_vld) act_sr1[ANN_HIDDEN_DIM*( ANN_WB_DW + INPUT_DW)-1:0] <= {ann_out,act_sr1[ANN_HIDDEN_DIM*( ANN_WB_DW + INPUT_DW)-1:(ANN_WB_DW + INPUT_DW)] }; // ??????, complete it after pe
                    else if (ann_shift == 2'b10) act_sr1[ANN_HIDDEN_DIM *( ANN_WB_DW + INPUT_DW)-1:0] <= {act_sr1[ANN_WB_DW + INPUT_DW-1:0 ],act_sr1[ANN_HIDDEN_DIM *( ANN_WB_DW + INPUT_DW)-1:( ANN_WB_DW + INPUT_DW) ]};
                    else;                    
                end
                else if (ann_mi_1|ann_mi_2) begin
                    if (ann_hidden_out_vld) act_sr1[ANN_HIDDEN_DIM*( ANN_WB_DW + FEATURE_SUM_DW)-1:0] <= {ann_out_mi,act_sr1[ANN_HIDDEN_DIM*( ANN_WB_DW + FEATURE_SUM_DW)-1:(ANN_WB_DW + FEATURE_SUM_DW)] }; // 
                    else if (ann_shift == 2'b10) act_sr1[ANN_HIDDEN_DIM *( ANN_WB_DW + FEATURE_SUM_DW)-1:0] <= {act_sr1[ANN_WB_DW + FEATURE_SUM_DW-1:0 ],act_sr1[ANN_HIDDEN_DIM *( ANN_WB_DW + FEATURE_SUM_DW)-1:( ANN_WB_DW + FEATURE_SUM_DW) ]};
                    else;                    
                end

            end
            else if (top_state == top_done) begin
                act_sr_init_done <= 0;
                act_sr1          <= 0;
            end
        end
    end
    // activations shift register buffer 1
    localparam  ENCODER_OUT_LEN = ENCODER_CHOUT * ENCODER_LENGTH_OUT;//TEMPT
    integer i;
    integer o;

    integer shift_id_2;
    integer shift_id_cnn12_2;
    integer shift_id_cnn22;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            act_sr2 <= 0;
        else begin
            if (top_state == seg_network) begin
                if (seg_state == encoder)  begin
                    if (encoder_out_vld)
                        act_sr2[ENCODER_OUT_LEN*DATA_OUT_DW-1:0] <= { encoder_out,  act_sr2[ENCODER_OUT_LEN*DATA_OUT_DW-1:DATA_OUT_DW] }; // encoder out
                    else
                        act_sr2 <= act_sr2 ;
                end
                else if (seg_state == lstm) begin
                    if (lstm_top_state == 3'd1) begin
                        if (lstm_xt_shift_en) begin

                            for (i = 0; i<ENCODER_CHOUT ; i = i + 1) begin // ..., CH3, CH2, CH1
                                act_sr2[(i+1)*ENCODER_LENGTH_OUT*DATA_OUT_DW-1-:ENCODER_LENGTH_OUT*DATA_OUT_DW] <= 
                                {act_sr2[(i*ENCODER_LENGTH_OUT+1)*DATA_OUT_DW -1-:DATA_OUT_DW], act_sr2[(i+1)*ENCODER_LENGTH_OUT*DATA_OUT_DW-1-:(ENCODER_LENGTH_OUT-1)*DATA_OUT_DW]}; // shift for input, xt
                            end                        
                        end
                        else
                            act_sr2 <= act_sr2;                    
                    end
                    else if (lstm_top_state == 3'd2) begin
                        if (lstm_xt_shift_en) begin
                            for (o = 0; o < ENCODER_CHOUT ; o = o + 1) begin // ..., CH3, CH2, CH1
                                act_sr2[(o+1)*ENCODER_LENGTH_OUT*DATA_OUT_DW-1-:ENCODER_LENGTH_OUT*DATA_OUT_DW] <= 
                                {act_sr2[((o+1)*ENCODER_LENGTH_OUT-1)*DATA_OUT_DW -1-: (ENCODER_LENGTH_OUT-1) * DATA_OUT_DW], act_sr2[((o+1)*ENCODER_LENGTH_OUT)*DATA_OUT_DW-1-:DATA_OUT_DW]};
                                
                            end                        
                        end
                        else
                            act_sr2 <= act_sr2; 
                    end
                    else if (lstm_top_state == 3'd3) begin // T64, T63, ..., T1
                        if (lstm_hidden_unit_vld)
                            act_sr2 <=  { lstm_hidden_cat, act_sr2[ACTIVATION_BUF_LEN2*DATA_OUT_DW-1:2*DATA_OUT_DW] };
                        else 
                            act_sr2 <= act_sr2;
                    end

                    else if (lstm_top_state == 3'd4) act_sr2<= act_sr2;
                    else act_sr2 <= act_sr2;

                end
                else if (seg_state == decoder) begin
                    if (decoder_top_state == dcnn1) begin
                        for (shift_id_2 = 0; shift_id_2 < PE_NUM; shift_id_2 = shift_id_2 + 1) begin
                            if ((shift_crl_all[2*shift_id_2-1+2-:2] == 1)& (cnt_bt_all[shift_id_2] == 0)) begin
                                // act_sr2 <= {act_sr2[shift_id_2*ENCODER_LENGTH_OUT*DATA_OUT_DW+DATA_OUT_DW-1-:DATA_OUT_DW], act_sr2[(shift_id_2+1)*ENCODER_LENGTH_OUT*DATA_OUT_DW-1-:(ENCODER_LENGTH_OUT-1)*DATA_OUT_DW]};
                                
                                act_sr2[0*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[1*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW+DATA_OUT_DW-1-:DATA_OUT_DW];
                                act_sr2[1*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[2*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[2*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[3*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[3*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[4*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[4*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[5*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[5*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[6*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[6*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[7*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[7*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[8*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[8*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[9*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[9*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[10*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[10*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[11*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[11*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[12*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[12*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[13*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[13*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[14*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[14*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[15*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[15*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[16*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[16*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[17*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[17*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[18*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[18*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[19*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[19*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[20*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[20*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[21*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[21*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[22*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[21*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[22*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[22*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[23*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[23*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[24*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[24*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[25*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[25*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[26*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[26*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[27*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[27*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[28*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[28*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[29*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[29*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[30*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[30*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[31*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[31*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[32*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[32*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[33*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[33*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[34*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[34*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[35*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[35*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[36*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[36*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[37*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[37*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[38*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[38*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[39*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[39*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[40*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[40*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[41*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[41*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[42*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[42*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[43*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[43*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[44*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[44*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[45*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[45*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[46*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[46*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[47*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[47*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[48*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[48*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[49*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[49*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[50*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[50*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[51*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[51*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[52*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[52*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[53*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[53*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[54*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[54*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[55*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[55*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[56*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[56*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[57*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[57*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[58*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[58*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[59*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[59*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[60*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[60*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[61*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[61*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[62*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[62*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[63*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]; 
                                act_sr2[63*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[0*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];                   
                            end
                            if ((shift_crl_all[2*shift_id_2+2-1-:2] == 2)& (cnt_bt_all[shift_id_2] == 0)) begin
                                act_sr2[0*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[3*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[1*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[4*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[2*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[5*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[3*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[6*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[4*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[7*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[5*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[8*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[6*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[9*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[7*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[10*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[8*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[11*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[9*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[12*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[10*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[13*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[11*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[14*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[12*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[15*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[13*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[16*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[14*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[17*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[15*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[18*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[16*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[19*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[17*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[20*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[18*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[21*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[19*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[22*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[20*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[23*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[21*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[24*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[22*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[25*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[23*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[26*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[24*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[27*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[25*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[28*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[26*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[29*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[27*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[30*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[28*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[31*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[29*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[32*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[30*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[33*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[31*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[34*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[32*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[35*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[33*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[36*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[34*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[37*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[35*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[38*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[36*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[39*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[37*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[40*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[38*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[41*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[39*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[42*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[40*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[43*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[41*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[44*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[42*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[45*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[43*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[46*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[44*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[47*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[45*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[48*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[46*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[49*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[47*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[50*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[48*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[51*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[49*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[52*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[50*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[53*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[51*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[54*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[52*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[55*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[53*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[56*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[54*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[57*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[55*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[58*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[56*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[59*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[57*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[60*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[58*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[61*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[59*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[62*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[60*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[63*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[61*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[0*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[62*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[1*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr2[63*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr2[2*LSTM_HS*DATA_OUT_DW + shift_id_2*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                            
                            end
                        end  
                    end 
                    else if (decoder_top_state == cnn11) begin
                        if ((decoder_out_vld) & ((cnt_cho_32 >CNN11_CHOUT/2-1)& (cnt_cho_32<CNN11_CHOUT))) begin
                            act_sr2 <= {decoder_out, act_sr2[CNN11_LENGTH_OUT/2*CNN11_CHOUT*DATA_OUT_DW-1:DATA_OUT_DW]};
                            
                        end
                        else act_sr2<=act_sr2;
                    end
                    else if (decoder_top_state == cnn12)    begin
                        for (shift_id_cnn12_2=0;shift_id_cnn12_2<CNN12_CHIN/2;shift_id_cnn12_2=shift_id_cnn12_2+1) begin
                            if  (shift_crl_all[2*(shift_id_cnn12_2+CNN12_CHIN/2)+2-1-:2] == 1)
                                act_sr2[(shift_id_cnn12_2+1)*CNN12_LENGTH_IN*DATA_OUT_DW-1-:CNN12_LENGTH_IN*DATA_OUT_DW] <= {act_sr2[shift_id_cnn12_2*CNN12_LENGTH_IN*DATA_OUT_DW+DATA_OUT_DW-1-:DATA_OUT_DW],act_sr2[(shift_id_cnn12_2+1)*CNN12_LENGTH_IN*DATA_OUT_DW-1-:(CNN12_LENGTH_IN-1)*DATA_OUT_DW] };
                            else if  (shift_crl_all[2*(shift_id_cnn12_2+CNN12_CHIN/2)+2-1-:2] == 2)
                                act_sr2[(shift_id_cnn12_2+1)*CNN12_LENGTH_IN*DATA_OUT_DW-1-:CNN12_LENGTH_IN*DATA_OUT_DW] <= {act_sr2[shift_id_cnn12_2*CNN12_LENGTH_IN*DATA_OUT_DW+4*DATA_OUT_DW-1-:4*DATA_OUT_DW],act_sr2[(shift_id_cnn12_2+1)*CNN12_LENGTH_IN*DATA_OUT_DW-1-:(CNN12_LENGTH_IN-4)*DATA_OUT_DW] };
                            else  act_sr2[(shift_id_cnn12_2+1)*CNN12_LENGTH_IN*DATA_OUT_DW-1-:CNN12_LENGTH_IN*DATA_OUT_DW]<= act_sr2[(shift_id_cnn12_2+1)*CNN12_LENGTH_IN*DATA_OUT_DW-1-:CNN12_LENGTH_IN*DATA_OUT_DW];

                        end                    
                    end  
                    else if (decoder_top_state == cnn21) begin
                        if (decoder_out_vld) begin
                            if (cnn22_is_first_2d) begin
                                act_sr2[CNN21_CHOUT*CNN21_LENGTH_OUT*DATA_OUT_DW/4-1:0] <= {decoder_out_cat[DATA_OUT_DW-1:0], act_sr2[CNN21_CHOUT/4*CNN21_LENGTH_OUT*DATA_OUT_DW-1:DATA_OUT_DW]};
                                act_sr2[CNN21_CHOUT*CNN21_LENGTH_OUT*DATA_OUT_DW/2-1:CNN21_CHOUT*CNN21_LENGTH_OUT*DATA_OUT_DW/4] <= {decoder_out_cat[2*DATA_OUT_DW-1:DATA_OUT_DW],act_sr2[CNN21_CHOUT*CNN21_LENGTH_OUT*DATA_OUT_DW/2-1:CNN21_CHOUT*CNN21_LENGTH_OUT*DATA_OUT_DW/4+DATA_OUT_DW ]};  
                            end
                            else begin
                                act_sr2[3*CNN21_CHOUT*CNN21_LENGTH_OUT*DATA_OUT_DW/4-1:CNN21_CHOUT*CNN21_LENGTH_OUT*DATA_OUT_DW/2] <= {decoder_out_cat[DATA_OUT_DW-1:0], act_sr2[3*CNN21_CHOUT/4*CNN21_LENGTH_OUT*DATA_OUT_DW-1:CNN21_CHOUT*CNN21_LENGTH_OUT*DATA_OUT_DW/2+DATA_OUT_DW ]};
                                act_sr2[CNN21_CHOUT*CNN21_LENGTH_OUT*DATA_OUT_DW-1:3*CNN21_CHOUT*CNN21_LENGTH_OUT*DATA_OUT_DW/4] <= {decoder_out_cat[2*DATA_OUT_DW-1:DATA_OUT_DW],act_sr2[CNN21_CHOUT*CNN21_LENGTH_OUT*DATA_OUT_DW-1:3*CNN21_CHOUT*CNN21_LENGTH_OUT*DATA_OUT_DW/4+DATA_OUT_DW ]};                            
                            end
                        end
                        else act_sr2<= act_sr2;
                    end      
                    else if (decoder_top_state == cnn22)   begin
                        for (shift_id_cnn22=0;shift_id_cnn22<CNN22_CHIN;shift_id_cnn22=shift_id_cnn22+1) begin
                            if (shift_crl_all[2*(shift_id_cnn22)+2-1-:2] == 1) 
                                act_sr2[(shift_id_cnn22+1)*CNN22_LENGTH_IN*DATA_OUT_DW-1-:CNN22_LENGTH_IN*DATA_OUT_DW] <= {act_sr2[shift_id_cnn22*CNN22_LENGTH_IN*DATA_OUT_DW+DATA_OUT_DW-1-:DATA_OUT_DW],act_sr2[(shift_id_cnn22+1)*CNN22_LENGTH_IN*DATA_OUT_DW-1-:(CNN22_LENGTH_IN-1)*DATA_OUT_DW] };
                            else if (shift_crl_all[2*(shift_id_cnn22)+2-1-:2] == 2)
                                act_sr2[(shift_id_cnn22+1)*CNN22_LENGTH_IN*DATA_OUT_DW-1-:CNN22_LENGTH_IN*DATA_OUT_DW] <= {act_sr2[shift_id_cnn22*CNN22_LENGTH_IN*DATA_OUT_DW+4*DATA_OUT_DW-1-:4*DATA_OUT_DW],act_sr2[(shift_id_cnn22+1)*CNN22_LENGTH_IN*DATA_OUT_DW-1-:(CNN22_LENGTH_IN-4)*DATA_OUT_DW] };
                            else act_sr2[(shift_id_cnn22+1)*CNN22_LENGTH_IN*DATA_OUT_DW-1-:CNN22_LENGTH_IN*DATA_OUT_DW]<=act_sr2[(shift_id_cnn22+1)*CNN22_LENGTH_IN*DATA_OUT_DW-1-:CNN22_LENGTH_IN*DATA_OUT_DW];
                        end                    
                    end
                end
                else begin
                    act_sr2 <= act_sr2 ;
                end
            end
            else if (top_state == post) begin 
                if (post_state == prepare_glb) act_sr2[INPUT_DW* ENCODER_LENGTH_IN-1:0] <= input_signal;
                else if (post_state == refine) begin
                    // if (refine_state == refine_idle)  begin
                    //     act_sr2[INPUT_DW* ENCODER_LENGTH_IN-1:0] <= input_signal;
                    // end
                    // else begin  
                        if (refine_shift)  act_sr2[INPUT_DW* ENCODER_LENGTH_IN-1:0] <= {  act_sr2[INPUT_DW-1:0], act_sr2[INPUT_DW* ENCODER_LENGTH_IN-1:INPUT_DW]}; //
                        else if (refine_shift_re) act_sr2[INPUT_DW* ENCODER_LENGTH_IN-1:0] <={ act_sr2[INPUT_DW* (ENCODER_LENGTH_IN-1)-1:0],act_sr2[INPUT_DW* ENCODER_LENGTH_IN-1-:INPUT_DW]}; 
                        
                        else act_sr2<= act_sr2;
                    // end
                end
                else if (post_state == embedding) begin
                    if (emb_shift) act_sr2[INPUT_DW* ENCODER_LENGTH_IN-1:0] <= {  act_sr2[INPUT_DW-1:0], act_sr2[INPUT_DW* ENCODER_LENGTH_IN-1:INPUT_DW]}; //
                    else act_sr2 <= act_sr2;
                end
                else if (post_state == post_done) act_sr2 <= 0;
                
            end
            // else if (top_state == ann) begin
            //     if (emb_shift) act_sr2[ANN_OUT_DIM*(INPUT_DW+ANN_WB_DW )-1:0] <= {ann_out,act_sr2[ANN_OUT_DIM*(INPUT_DW+ANN_WB_DW )-1:(INPUT_DW+ANN_WB_DW )]}; 
            //     else act_sr2<= act_sr2;
            // end
            else if (top_state == top_done) act_sr2 <= 0;
            else;
        end
    end
    integer shift_id_3;
    integer  shift_id_dcnn2;

    // wire [QRS_EMB_LEN*EMB_DW-1:0] act_sr3_qrs_emb_buffer;
    // assign act_sr3_qrs_emb_buffer = act_sr3[NUM_FEAS_MI * (NUM_LEADS+0) * FEATURE_SUM_DW + DIR_DW * NUM_LEADS + NUM_LEADS * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW  -1-:QRS_EMB_LEN*EMB_DW];
    // wire [T_EMB_LEN*EMB_DW-1:0] act_sr3_t_emb_buffer;
    // assign act_sr3_t_emb_buffer = act_sr3[NUM_FEAS_MI * (NUM_LEADS+0) * FEATURE_SUM_DW + DIR_DW * NUM_LEADS + (NUM_LEADS-1) * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW + T_EMB_LEN  * EMB_DW -1-:T_EMB_LEN*EMB_DW];
    // wire [DIR_DW-1:0] act_sr3_t_dir;
    // assign act_sr3_t_dir = act_sr3[NUM_FEAS_MI * (NUM_LEADS+0) * FEATURE_SUM_DW + DIR_DW * NUM_LEADS + (NUM_LEADS-1) * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW -1-:DIR_DW];
    // wire [FEATURE_SUM_DW-1:0] act_sr3_st_slo;
    // assign act_sr3_st_slo = act_sr3[NUM_FEAS_MI * (NUM_LEADS+0) * FEATURE_SUM_DW + DIR_DW * (NUM_LEADS-1) + (NUM_LEADS-1) * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW -1-: FEATURE_SUM_DW];
    // wire [FEATURE_SUM_DW-1:0] act_sr3_st_amp_iso;
    // assign act_sr3_st_amp_iso = act_sr3[NUM_FEAS_MI * (NUM_LEADS+0) * FEATURE_SUM_DW + DIR_DW * (NUM_LEADS-1) + (NUM_LEADS-1) * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW -FEATURE_SUM_DW -1-: FEATURE_SUM_DW];
    // wire [FEATURE_SUM_DW-1:0] act_sr3_r_amp_iso;
    // assign act_sr3_r_amp_iso = act_sr3[NUM_FEAS_MI * (NUM_LEADS+0) * FEATURE_SUM_DW + DIR_DW * (NUM_LEADS-1) + (NUM_LEADS-1) * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW-2*FEATURE_SUM_DW -1-: FEATURE_SUM_DW];
    // wire [FEATURE_SUM_DW-1:0] act_sr3_t_amp_iso;
    // assign act_sr3_t_amp_iso = act_sr3[NUM_FEAS_MI * (NUM_LEADS+0) * FEATURE_SUM_DW + DIR_DW * (NUM_LEADS-1) + (NUM_LEADS-1) * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW -3*FEATURE_SUM_DW-1-: FEATURE_SUM_DW];
    // wire  [FEATURE_SUM_DW-1:0] act_sr3_s_amp_iso;
    // assign act_sr3_s_amp_iso = act_sr3[NUM_FEAS_MI * (NUM_LEADS+0) * FEATURE_SUM_DW + DIR_DW * (NUM_LEADS-1) + (NUM_LEADS-1) * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW -4*FEATURE_SUM_DW-1-: FEATURE_SUM_DW];
    // wire  [FEATURE_SUM_DW-1:0] act_sr3_q_amp_iso;
    // assign act_sr3_q_amp_iso = act_sr3[NUM_FEAS_MI * (NUM_LEADS+0) * FEATURE_SUM_DW + DIR_DW * (NUM_LEADS-1) + (NUM_LEADS-1) * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW-5*FEATURE_SUM_DW -1-: FEATURE_SUM_DW];
    // wire  [FEATURE_SUM_DW-1:0] act_sr3_r_amp_t_amp;
    // assign act_sr3_r_amp_t_amp = act_sr3[NUM_FEAS_MI * (NUM_LEADS+0) * FEATURE_SUM_DW + DIR_DW * (NUM_LEADS-1) + (NUM_LEADS-1) * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW -6*FEATURE_SUM_DW-1-: FEATURE_SUM_DW];
    
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            act_sr3          <= 0;
        end
        else begin
            if (top_state == seg_network) begin
                if (seg_state == lstm) begin

                    if (lstm_top_state == 3'd2) begin
                        if (lstm_hidden_unit_vld) //T1, T2, T3, ..., T64
                            act_sr3 <= { lstm_hidden_cat, act_sr3[ACTIVATION_BUF_LEN3*DATA_OUT_DW-1:2*DATA_OUT_DW] };
                        else
                            act_sr3 <=  act_sr3;   
                    end
                    else if (lstm_top_state == 3'd3) begin // <-
                        if (lstm_xt_shift_en)
                            act_sr3 <= {act_sr3[(ACTIVATION_BUF_LEN3-LSTM_HS)*DATA_OUT_DW-1:0],act_sr3[DATA_OUT_DW*ACTIVATION_BUF_LEN3-1-:DATA_OUT_DW*LSTM_HS]};
                        else
                            act_sr3 <= act_sr3;
                    end
                    else if (lstm_top_state == 3'd4) begin //->, T1, T2, T3, T4,...,T64
                        if (lstm_hidden_unit_vld)
                            act_sr3 <= { lstm_hidden_cat, act_sr3[ACTIVATION_BUF_LEN3*DATA_OUT_DW-1:2*DATA_OUT_DW] };
                            
                        else
                            act_sr3 <= act_sr3;                    
                    end
                    else begin
                        act_sr3 <=  act_sr3; 
                    end
                end
                else if (seg_state == decoder) begin
                    if (decoder_top_state == dcnn1) begin
                        for (shift_id_3 = 0; shift_id_3 < PE_NUM; shift_id_3 = shift_id_3 + 1) begin
                            if ((shift_crl_all[2*shift_id_3+2-1-:2] == 1)& (cnt_bt_all[shift_id_3] == 1)) begin
                                act_sr3[63*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[62*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[62*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[61*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[61*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[60*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[60*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[59*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[59*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[58*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[58*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[57*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[57*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[56*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[56*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[55*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[55*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[54*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[54*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[53*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[53*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[52*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[52*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[51*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[51*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[50*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[50*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[49*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[49*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[48*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[48*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[47*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[47*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[46*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[46*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[45*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[45*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[44*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[44*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[43*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[43*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[42*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[42*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[41*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[41*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[40*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[40*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[39*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[39*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[38*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[38*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[37*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[37*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[36*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[36*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[35*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[35*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[34*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[34*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[33*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[33*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[32*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[32*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[31*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[31*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[30*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[30*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[29*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[29*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[28*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[28*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[27*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[27*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[26*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[26*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[25*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[25*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[24*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[24*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[23*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[23*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[22*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[22*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[21*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[21*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[20*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[20*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[19*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[19*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[18*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[18*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[17*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[17*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[16*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[16*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[15*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[15*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[14*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[14*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[13*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[13*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[12*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[12*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[11*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[11*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[10*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[10*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[9*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[9*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[8*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[8*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[7*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[7*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[6*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[6*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[5*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[5*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[4*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[4*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[3*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[3*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[2*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[2*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[1*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[1*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[0*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[0*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[63*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                            end
                            if ((shift_crl_all[2*shift_id_3+2-1-:2] == 2)& (cnt_bt_all[shift_id_3] == 1))begin
                                act_sr3[63*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[60*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[62*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[59*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[61*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[58*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[60*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[57*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[59*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[56*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[58*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[55*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[57*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[54*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[56*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[53*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[55*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[52*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[54*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[51*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[53*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[50*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[52*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[49*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[51*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[48*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[50*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[47*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[49*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[46*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[48*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[45*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[47*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[44*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[46*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[43*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[45*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[42*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[44*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[41*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];                                            
                                act_sr3[43*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[40*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[42*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[39*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[41*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[38*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[40*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[37*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[39*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[36*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[38*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[35*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[37*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[34*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[36*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[33*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[35*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[32*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[34*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[31*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[33*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[30*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[32*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[29*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[31*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[28*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[30*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[27*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[29*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[26*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[28*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[25*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[27*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[24*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[26*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[23*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[25*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[22*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[24*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[21*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[23*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[20*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[22*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[19*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[21*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[18*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[20*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[17*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[19*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[16*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[18*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[15*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[17*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[14*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[16*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[13*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[15*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[12*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[14*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[11*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[13*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[10*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[12*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[9*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[11*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[8*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[10*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[7*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[9*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[6*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[8*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[5*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[7*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[4*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[6*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[3*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[5*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[2*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[4*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[1*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[3*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[0*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[2*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[63*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[1*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[62*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                                act_sr3[0*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW]<= act_sr3[61*LSTM_HS*DATA_OUT_DW + shift_id_3*DATA_OUT_DW-1+DATA_OUT_DW-:DATA_OUT_DW];
                            end
                        end
                    end
                    else if (decoder_top_state == cnn12) begin
                        if (decoder_out_vld) begin
                            act_sr3 <= {decoder_out, act_sr3[CNN12_LENGTH_OUT*CNN12_CHOUT*DATA_OUT_DW-1:DATA_OUT_DW]};
                        end
                        else act_sr3 <= act_sr3;
                    end
                    else if (decoder_top_state == dcnn2) begin
                        for (shift_id_dcnn2 = 0; shift_id_dcnn2<DCNN2_CHIN; shift_id_dcnn2= shift_id_dcnn2+ 1) begin
                            if (shift_crl_all[2*(shift_id_dcnn2)+2-1-:2] == 1) begin // CHIN < PE_NUM
                                act_sr3 [(shift_id_dcnn2+1)*DCNN2_LENGTH_IN*DATA_OUT_DW-1-:DCNN2_LENGTH_IN*DATA_OUT_DW]<={act_sr3[shift_id_dcnn2*DCNN2_LENGTH_IN*DATA_OUT_DW+DATA_OUT_DW-1-:DATA_OUT_DW] ,act_sr3[(shift_id_dcnn2+1)*DCNN2_LENGTH_IN*DATA_OUT_DW-1-:(DCNN2_LENGTH_IN-1)*DATA_OUT_DW]};
                            end
                            else if (shift_crl_all[2*shift_id_dcnn2+2-1-:2] == 2)
                                act_sr3 [(shift_id_dcnn2+1)*DCNN2_LENGTH_IN*DATA_OUT_DW-1-:DCNN2_LENGTH_IN*DATA_OUT_DW] <= {act_sr3[shift_id_dcnn2*DCNN2_LENGTH_IN*DATA_OUT_DW+3*DATA_OUT_DW-1-:3*DATA_OUT_DW],act_sr3[(shift_id_dcnn2+1)*DCNN2_LENGTH_IN*DATA_OUT_DW-1-:(DCNN2_LENGTH_IN-3)*DATA_OUT_DW]};
                            
                            else act_sr3[(shift_id_dcnn2+1)*DCNN2_LENGTH_IN*DATA_OUT_DW-1-:DCNN2_LENGTH_IN*DATA_OUT_DW] <= act_sr3[(shift_id_dcnn2+1)*DCNN2_LENGTH_IN*DATA_OUT_DW-1-:DCNN2_LENGTH_IN*DATA_OUT_DW];
                        end
                    end
                end
            end
            else if (top_state == feature_map) begin
                if (save_fea_en ==  2'b10) begin // save for mi
                    if (cnt_lead != NUM_LEADS -1) begin
                        act_sr3[ NUM_FEAS_MI * (NUM_LEADS+1) * FEATURE_SUM_DW + DIR_DW * NUM_LEADS + NUM_LEADS * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW -1 : 0] 
                                <= {qrs_emb_buffer,
                                    t_emb_buffer,
                                    t_dir,
                                    {{(FEATURE_SUM_DW - INPUT_DW){st_slo[INPUT_DW-1]}},st_slo}, 
                                    {{(FEATURE_SUM_DW - INPUT_DW){st_amp_iso[INPUT_DW-1]}},st_amp_iso},
                                    {{(FEATURE_SUM_DW - INPUT_DW){t_amp_iso[INPUT_DW-1]}},t_amp_iso}, 
                                    {{(FEATURE_SUM_DW - INPUT_DW){s_amp_iso[INPUT_DW-1]}},s_amp_iso},
                                    {{(FEATURE_SUM_DW - INPUT_DW){q_amp_iso[INPUT_DW-1]}},q_amp_iso},
                                    act_sr3[NUM_FEAS_MI * (NUM_LEADS+1) * FEATURE_SUM_DW + DIR_DW * NUM_LEADS + NUM_LEADS * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW  -1 :  NUM_FEAS_MI * FEATURE_SUM_DW +(QRS_EMB_LEN + T_EMB_LEN) * EMB_DW + DIR_DW]};
                    end
                    else  begin
                        act_sr3[NUM_FEAS_MI * (NUM_LEADS+1) * FEATURE_SUM_DW + DIR_DW * NUM_LEADS + NUM_LEADS * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW  -1 : 0] 
                                <= {st_slo_sum,
                                    st_amp_iso_sum,
                                    t_amp_iso_sum,
                                    s_amp_iso_sum,
                                    q_amp_iso_sum,
                                    qrs_emb_buffer,
                                    t_emb_buffer,
                                    t_dir,            
                                    {{(FEATURE_SUM_DW - INPUT_DW){st_slo[INPUT_DW-1]}},st_slo}, 
                                    {{(FEATURE_SUM_DW - INPUT_DW){st_amp_iso[INPUT_DW-1]}},st_amp_iso},
                                    {{(FEATURE_SUM_DW - INPUT_DW){t_amp_iso[INPUT_DW-1]}},t_amp_iso},
                                    {{(FEATURE_SUM_DW - INPUT_DW){s_amp_iso[INPUT_DW-1]}},s_amp_iso},
                                    {{(FEATURE_SUM_DW - INPUT_DW){q_amp_iso[INPUT_DW-1]}},q_amp_iso},
                                    act_sr3[NUM_FEAS_MI * (NUM_LEADS+1) * FEATURE_SUM_DW + DIR_DW * NUM_LEADS + NUM_LEADS * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW  -1 : 2*NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW + DIR_DW ]};                        
                    end
                end

                else;
            end
            else if (top_state == ann) begin
                if (feature_shift) begin // fix sum features
                        act_sr3[NUM_FEAS_MI * (NUM_LEADS+1) * FEATURE_SUM_DW + DIR_DW * NUM_LEADS + NUM_LEADS * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW  -1 : 0 ]
                        <= {act_sr3[NUM_FEAS_MI * (NUM_LEADS+1) * FEATURE_SUM_DW + DIR_DW * NUM_LEADS + NUM_LEADS * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW-1-:NUM_FEAS_MI*FEATURE_SUM_DW],
                        act_sr3[NUM_FEAS_MI*FEATURE_SUM_DW +DIR_DW + (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW-1:0],
                        act_sr3[NUM_FEAS_MI * NUM_LEADS * FEATURE_SUM_DW + DIR_DW * NUM_LEADS + NUM_LEADS * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW-1:NUM_FEAS_MI*FEATURE_SUM_DW +DIR_DW + (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW]};
                end
                else;
            end
            else if (top_state == top_done) begin
                act_sr3          <= 0 ; // cannot be reset
            end
            else begin
                act_sr3 <=  act_sr3; 
            end
        end
    end 

    integer  shift_id_cnn11;


    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            act_sr4          <= 0;
        end    
        else begin
            if (top_state == seg_network) begin
                if (seg_state == decoder) begin
                    if (decoder_top_state == dcnn1) begin
                        if (decoder_out_vld) begin
                            act_sr4 <= {decoder_out, act_sr4[ACTIVATION_BUF_LEN4*DATA_OUT_DW-1:DATA_OUT_DW]};
                        end
                        else begin
                            act_sr4 <= act_sr4;
                        end
                    end
                    else if (decoder_top_state == cnn11) begin

                        for (shift_id_cnn11 = 0;shift_id_cnn11<PE_NUM;shift_id_cnn11 = shift_id_cnn11+1 ) begin
                            if (shift_crl_all[2*shift_id_cnn11+2-1-:2] == 1) begin
                                act_sr4 [(shift_id_cnn11+1)*DCNN1_LENGTH_OUT*DATA_OUT_DW-1-:DCNN1_LENGTH_OUT*DATA_OUT_DW] <= {act_sr4[shift_id_cnn11*DCNN1_LENGTH_OUT*DATA_OUT_DW+DATA_OUT_DW-1-:DATA_OUT_DW],act_sr4[(shift_id_cnn11+1)*DCNN1_LENGTH_OUT*DATA_OUT_DW-1-:(DCNN1_LENGTH_OUT-1)*DATA_OUT_DW]};
                            end
                            else if (shift_crl_all[2*shift_id_cnn11+2-1-:2] == 2)
                                act_sr4 [(shift_id_cnn11+1)*DCNN1_LENGTH_OUT*DATA_OUT_DW-1-:DCNN1_LENGTH_OUT*DATA_OUT_DW] <= {act_sr4[shift_id_cnn11*DCNN1_LENGTH_OUT*DATA_OUT_DW+4*DATA_OUT_DW-1-:4*DATA_OUT_DW],act_sr4[(shift_id_cnn11+1)*DCNN1_LENGTH_OUT*DATA_OUT_DW-1-:(DCNN1_LENGTH_OUT-4)*DATA_OUT_DW]};
                            else act_sr4 [(shift_id_cnn11+1)*DCNN1_LENGTH_OUT*DATA_OUT_DW-1-:DCNN1_LENGTH_OUT*DATA_OUT_DW] <= act_sr4 [(shift_id_cnn11+1)*DCNN1_LENGTH_OUT*DATA_OUT_DW-1-:DCNN1_LENGTH_OUT*DATA_OUT_DW];
                        end
                    end
                end 
                else if (seg_state == seg_done)   begin
                    act_sr4 <= 0;
                end   
                else;
            
            end
            else if (top_state == ann) begin
                if (ann_state ==  5'd1) begin // ann1
                    if (input_init_en) act_sr4[FEATURE_DIM*INPUT_DW-1:0] <= feature_matrix;
                    else if (ann_shift == 2'b01) act_sr4[3*SPAD_DEPTH *INPUT_DW-1:0] <= {act_sr4[SPAD_DEPTH *INPUT_DW-1:0 ],act_sr4[3*SPAD_DEPTH *INPUT_DW-1:SPAD_DEPTH *INPUT_DW ]};
                    else;
                end
                else if (ann_mi_1) begin//feature_matrix_mi[2*NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW + DIR_DW -1 -:NUM_FEAS_MI  * FEATURE_SUM_DW],
                    if (input_init_en) act_sr4[FEATURE_DIM_MI*FEATURE_SUM_DW-1:0] <= 
                                        {feature_matrix_mi[2*NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW + DIR_DW -1 -:NUM_FEAS_MI  * FEATURE_SUM_DW],
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 0) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-0) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 1) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-1) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 2) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-2) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 3) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-3) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 4) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-4) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 5) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-5) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 6) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-6) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 7) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-7) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 8) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-8) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 9) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-9) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 10) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-10) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 11) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-11) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 12) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-12) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 13) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-13) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 14) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-14) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 15) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-15) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 16) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-16) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 17) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-17) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 18) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-18) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 19) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-19) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 20) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-20) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 21) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-21) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 22) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-22) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 23) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-23) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 24) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-24) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 25) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-25) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 26) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-26) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 27) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-27) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 28) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-28) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 29) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-29) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 30) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-30) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 31) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-31) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 32) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-32) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 33) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-33) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 34) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-34) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 35) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-35) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 36) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-36) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 37) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-37) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 38) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-38) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 39) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-39) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 40) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-40) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 41) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-41) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 42) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-42) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 43) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-43) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 44) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-44) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 45) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-45) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 46) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-46) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 47) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-47) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 48) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-48) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 49) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-49) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 50) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-50) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 51) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-51) * EMB_DW + DIR_DW -1 -: EMB_DW]},
                                        {{(FEATURE_SUM_DW-EMB_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN - 52) * EMB_DW + DIR_DW -1]}},feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN-52) * EMB_DW + DIR_DW -1 -: EMB_DW]},                                        
                                        {{(FEATURE_SUM_DW-DIR_DW){feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + DIR_DW -1 ]}},feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW + DIR_DW -1 -: DIR_DW]},
                                        feature_matrix_mi[ NUM_FEAS_MI * FEATURE_SUM_DW -1 -: NUM_FEAS_MI * FEATURE_SUM_DW]};
                    else if (ann_shift == 2'b10) act_sr4[FEATURE_DIM_MI * FEATURE_SUM_DW-1:0] <= {act_sr4[FEATURE_SUM_DW-1: 0],act_sr4[FEATURE_DIM_MI * FEATURE_SUM_DW-1:FEATURE_SUM_DW ]};
                    else;                    
                end
            end
            else if (top_state == top_done) begin
                act_sr4 <= 0;
            end
        end
    end


    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            feature_rb <= 0;
        end
        else begin
            if (top_state == feature_map) begin
                if (save_fea_en ==  2'b11) begin
                    feature_rb[SAVE_NUM_BEATS * INTEVAL_DW-1 : 0] <= {rr_pre_d, feature_rb[SAVE_NUM_BEATS * INTEVAL_DW-1 : INTEVAL_DW]};
                    feature_rb[2*SAVE_NUM_BEATS * INTEVAL_DW-1 : SAVE_NUM_BEATS * INTEVAL_DW] <= {qrs_d,feature_rb[2*SAVE_NUM_BEATS * INTEVAL_DW-1 : SAVE_NUM_BEATS * INTEVAL_DW+INTEVAL_DW] };
                    feature_rb[SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : 2*SAVE_NUM_BEATS * INTEVAL_DW] <= {r_amp_d, feature_rb[SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : 2*SAVE_NUM_BEATS * INTEVAL_DW +INPUT_DW]};
                    feature_rb[2*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : SAVE_NUM_BEATS*INPUT_DW + 2*SAVE_NUM_BEATS * INTEVAL_DW] <= {q_amp_d, feature_rb[2*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : SAVE_NUM_BEATS*INPUT_DW + 2*SAVE_NUM_BEATS * INTEVAL_DW +INPUT_DW]};
                    feature_rb[3*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : 2*SAVE_NUM_BEATS*INPUT_DW + 2*SAVE_NUM_BEATS * INTEVAL_DW] <= {s_amp_d, feature_rb[3*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : 2*SAVE_NUM_BEATS*INPUT_DW + 2*SAVE_NUM_BEATS * INTEVAL_DW +INPUT_DW]};
                    feature_rb[4*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : 3*SAVE_NUM_BEATS*INPUT_DW + 2*SAVE_NUM_BEATS * INTEVAL_DW] <= {p_amp_d, feature_rb[4*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : 3*SAVE_NUM_BEATS*INPUT_DW + 2*SAVE_NUM_BEATS * INTEVAL_DW +INPUT_DW]};
                    feature_rb[5*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : 4*SAVE_NUM_BEATS*INPUT_DW + 2*SAVE_NUM_BEATS * INTEVAL_DW] <= {t_amp_d, feature_rb[5*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : 4*SAVE_NUM_BEATS*INPUT_DW + 2*SAVE_NUM_BEATS * INTEVAL_DW +INPUT_DW]};
                end
                else if (save_fea_en ==  2'b01) begin
                    feature_rb[SAVE_NUM_BEATS * INTEVAL_DW-1 : 0] <= {rr_pre, feature_rb[SAVE_NUM_BEATS * INTEVAL_DW-1 : INTEVAL_DW]};
                    feature_rb[2*SAVE_NUM_BEATS * INTEVAL_DW-1 : SAVE_NUM_BEATS * INTEVAL_DW] <= {qrs,feature_rb[2*SAVE_NUM_BEATS * INTEVAL_DW-1 : SAVE_NUM_BEATS * INTEVAL_DW+INTEVAL_DW] };
                    feature_rb[SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : 2*SAVE_NUM_BEATS * INTEVAL_DW] <= {r_amp, feature_rb[SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : 2*SAVE_NUM_BEATS * INTEVAL_DW +INPUT_DW]};
                    feature_rb[2*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : SAVE_NUM_BEATS*INPUT_DW + 2*SAVE_NUM_BEATS * INTEVAL_DW] <= {q_amp, feature_rb[2*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : SAVE_NUM_BEATS*INPUT_DW + 2*SAVE_NUM_BEATS * INTEVAL_DW +INPUT_DW]};
                    feature_rb[3*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : 2*SAVE_NUM_BEATS*INPUT_DW + 2*SAVE_NUM_BEATS * INTEVAL_DW] <= {s_amp, feature_rb[3*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : 2*SAVE_NUM_BEATS*INPUT_DW + 2*SAVE_NUM_BEATS * INTEVAL_DW +INPUT_DW]};
                    feature_rb[4*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : 3*SAVE_NUM_BEATS*INPUT_DW + 2*SAVE_NUM_BEATS * INTEVAL_DW] <= {p_amp, feature_rb[4*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : 3*SAVE_NUM_BEATS*INPUT_DW + 2*SAVE_NUM_BEATS * INTEVAL_DW +INPUT_DW]};
                    feature_rb[5*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : 4*SAVE_NUM_BEATS*INPUT_DW + 2*SAVE_NUM_BEATS * INTEVAL_DW] <= {t_amp, feature_rb[5*SAVE_NUM_BEATS*INPUT_DW+2*SAVE_NUM_BEATS * INTEVAL_DW-1 : 4*SAVE_NUM_BEATS*INPUT_DW + 2*SAVE_NUM_BEATS * INTEVAL_DW +INPUT_DW]};
                     
                end
                else;
            end
            else if (top_state == top_done) begin
                feature_rb          <= feature_rb ; // cannot be reset
            end
            else begin
                feature_rb <=  feature_rb; 
            end
        end
    end
endmodule