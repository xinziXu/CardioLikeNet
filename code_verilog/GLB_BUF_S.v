`timescale  1ns/100ps
module GLB_BUF_S #(parameter  INPUT_DW = 12,
    DATA_OUT_DW = 8,
    FEATURE_SUM_DW = INPUT_DW + 4,
    ENCODER_LENGTH_IN = 256,
    INTEVAL_DW= $clog2(ENCODER_LENGTH_IN+1),
    SAVE_NUM_BEATS = 9,
    NUM_FEAS_MI = 5,
    NUM_LEADS = 12,
    DATA_BQ_DW = 32,
    SPAD_DEPTH = 8,
    DCNN1_LENGTH_OUT = ENCODER_LENGTH_IN/2,
    CNN22_LENGTH_OUT =  ENCODER_LENGTH_IN,
    DIR_DW = 2,
    LABEL_DW = 2,
    EMB_DW = 2,
    QRS_EMB_LEN = 24,
    T_EMB_LEN = 25,
    ANN_WB_DW = 16,
    FEATURE_DIM = 22,
    FEATURE_DIM_MI = 64,
    ANN_HIDDEN_DIM =32,
    ACTIVATION_BUF_LEN1 = (DCNN1_LENGTH_OUT-2)*DATA_BQ_DW,
    ACTIVATION_BUF_LEN2 = INPUT_DW*ENCODER_LENGTH_IN,
    ACTIVATION_BUF_LEN3 =  NUM_FEAS_MI * (NUM_LEADS+1) * FEATURE_SUM_DW + DIR_DW * NUM_LEADS + NUM_LEADS * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW,//change width
    ACTIVATION_BUF_LEN4 =  FEATURE_DIM_MI*FEATURE_SUM_DW)
    (input wclk,
    input rst_n,
    input [INPUT_DW*ENCODER_LENGTH_IN-1:0] input_signal,
    // input [LABEL_DW*ENCODER_LENGTH_IN-1:0] softmax_out_all,
    // input [DATA_OUT_DW*CNN21_OUT-1:0] cnn21_out,
    input [3:0] top_state,
    input [3:0] seg_state,
    input [2:0] decoder_top_state,
    input [4:0] ann_state,
    input cnn22_is_first_2d,  
    input [DATA_BQ_DW-1: 0] pe_out_32b_1, // change width

    input [LABEL_DW-1:0] softmax_out,
    input decoder_out_vld,
    input dcnn1_temp_value_vld,
    input dcnn1_transfer_temp_value_en,
    input dcnn1_temp_rst,
    input [$clog2(NUM_LEADS+1)-1 : 0] cnt_lead,
    output reg [ACTIVATION_BUF_LEN1-1:0] act_sr1, //change width
    output reg [ACTIVATION_BUF_LEN2-1:0] act_sr2,//change width
    output reg [ACTIVATION_BUF_LEN3-1:0] act_sr3,//change width
    output reg [ACTIVATION_BUF_LEN4-1:0] act_sr4,//change width
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

    input signed [INPUT_DW - 1: 0] q_amp_iso,
    input signed [INPUT_DW - 1: 0] s_amp_iso,
    input signed [INPUT_DW - 1: 0] t_amp_iso,
    // input signed [INPUT_DW - 1: 0] r_amp_iso,
    input signed [INPUT_DW - 1: 0] st_amp_iso,
    input signed [INPUT_DW - 1: 0] st_slo,
    input signed [FEATURE_SUM_DW - 1: 0] q_amp_iso_sum,
    input signed [FEATURE_SUM_DW - 1: 0] s_amp_iso_sum,
    input signed [FEATURE_SUM_DW - 1: 0] t_amp_iso_sum,
    // input signed [FEATURE_SUM_DW - 1: 0] r_amp_iso_sum,
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
    

    localparam cnn22 = 3'b010;
    localparam dcnn1 =  3'b001; //from sram

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
        end
        else begin
            if (top_state == seg_network) begin
                if (decoder_top_state == dcnn1) begin
                        if (dcnn1_temp_value_vld) begin
                            act_sr1 [(DCNN1_LENGTH_OUT-2)*DATA_BQ_DW-1:0] <= {pe_out_32b_1,act_sr1[(DCNN1_LENGTH_OUT-2)*DATA_BQ_DW-1:DATA_BQ_DW]};
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
                else if (decoder_top_state == cnn22)   begin
                    if(decoder_out_vld) begin
                        if (!cnn22_is_first_2d) begin
                            act_sr1[LABEL_DW*CNN22_LENGTH_OUT-1:0] <= {softmax_out, act_sr1[LABEL_DW*CNN22_LENGTH_OUT-1:LABEL_DW]};                        
                            end
                        else act_sr1 <= act_sr1;
                    end
                    else act_sr1 <= act_sr1;
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
                act_sr1          <= 0;
            end
        end
    end
    // activations shift register buffer 1
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            act_sr2 <= 0;
        else begin
            if (top_state == post) begin 
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

            else if (top_state == top_done) begin
                act_sr2          <= 0 ; // cannot be reset
            end
            else begin
                act_sr2 <=  act_sr2; 
            end
            // else if (top_state == ann) begin
            //     if (emb_shift) act_sr2[ANN_OUT_DIM*(INPUT_DW+ANN_WB_DW )-1:0] <= {ann_out,act_sr2[ANN_OUT_DIM*(INPUT_DW+ANN_WB_DW )-1:(INPUT_DW+ANN_WB_DW )]}; 
            //     else act_sr2<= act_sr2;
            // end
            // else if (top_state == top_done) act_sr2 <= 0;
            // else;
        end
    end

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            act_sr3 <= 0;
        end
        else begin
            if (top_state == feature_map) begin
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
        end
    end

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            act_sr4          <= 0;
        end    
        else begin
            if (top_state == ann) begin
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