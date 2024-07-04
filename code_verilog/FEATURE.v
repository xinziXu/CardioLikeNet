`timescale  1ns/100ps
module FEATURE #(parameter INPUT_DW = 12,
    DATA_DW = 8,
    LENGTH_IN = 256,
    ARR_LABEL_DW = 2,
    INIT_NUM_BEATS = 8,
    NUM_FEAS_MI = 5,
    INTEVAL_DW = $clog2(LENGTH_IN+1),
    // ACTIVATION_BUF_LEN1 = 32*64,
    NUM_LEADS = 12,
    FEATURE_SUM_DW = INPUT_DW + 4)
    (  input wclk,
    input rst_n,  
    input feature_rdy,
    input [5*(INIT_NUM_BEATS+1)*INPUT_DW+2*(INIT_NUM_BEATS+1) * INTEVAL_DW-1:0] feature_rb,
    input [INTEVAL_DW-1:0] rr_pre,
    input [INTEVAL_DW-1:0] rr_post,
    input signed [INPUT_DW-1:0] r_amp,
    input signed [INPUT_DW-1:0] t_amp,
    input signed [INPUT_DW-1:0] p_amp,
    input signed [INPUT_DW-1:0] q_amp,
    input [INTEVAL_DW-1:0] q_loc,
    input [INTEVAL_DW-1:0] s_loc,
    input signed [INPUT_DW-1:0] s_amp,
    input [INTEVAL_DW-1:0] r_loc,
    input [$clog2(LENGTH_IN+1)-1:0] t_loc,
    input [$clog2(LENGTH_IN+1)-1:0] st_loc,
    input signed [INPUT_DW-1:0] st_amp,
    input signed [INPUT_DW-1:0] st_amp_1,
    input signed [INPUT_DW-1:0] st_amp_2,
    input signed [INPUT_DW-1:0] st_amp_4,
    input signed [INPUT_DW-1:0] st_amp_6,
    input signed [INPUT_DW-1:0] iso_line,
    input [ARR_LABEL_DW-1:0] predict_pre, // 0  norm
    output reg [1:0] save_fea_en,
    
    output reg  signed [INTEVAL_DW-1:0] rr_diff,
    output reg [INTEVAL_DW-1:0] qrs,
    output reg signed [INTEVAL_DW -1 : 0] rr_pre_rr_ave,
    output reg signed [INTEVAL_DW -1 : 0] rr_post_rr_ave,
    output reg signed [INTEVAL_DW -1 : 0] qrs_cur_qrs_ave,
    output reg signed [INPUT_DW - 1: 0] r_amp_r_amp_ave,
    output reg signed [INPUT_DW - 1: 0] q_amp_q_amp_ave,
    output reg signed [INPUT_DW - 1: 0] s_amp_s_amp_ave,
    output reg signed [INPUT_DW - 1: 0] p_amp_p_amp_ave,
    output reg signed [INPUT_DW - 1: 0] t_amp_t_amp_ave,
    // output reg signed [INPUT_DW - 1: 0] r_amp_t_amp,
    output reg signed [INPUT_DW - 1: 0] q_amp_iso,
    output reg signed [INPUT_DW - 1: 0] s_amp_iso,
    output reg signed [INPUT_DW - 1: 0] t_amp_iso,
    // output reg signed [INPUT_DW - 1: 0] r_amp_iso,
    output reg signed [INPUT_DW - 1: 0] st_amp_iso,
    output reg signed [INPUT_DW - 1: 0] st_slo,
    // output reg signed [FEATURE_SUM_DW - 1: 0] r_amp_t_amp_sum,
    output reg signed [FEATURE_SUM_DW - 1: 0] q_amp_iso_sum,
    output reg signed [FEATURE_SUM_DW - 1: 0] s_amp_iso_sum,
    output reg signed [FEATURE_SUM_DW - 1: 0] t_amp_iso_sum,
    // output reg signed [FEATURE_SUM_DW - 1: 0] r_amp_iso_sum,
    output reg signed [FEATURE_SUM_DW - 1: 0] st_amp_iso_sum,
    output reg signed [FEATURE_SUM_DW - 1: 0] st_slo_sum,
    output feature_done,
   
    output init_features_end,
    input mode,
    input [$clog2(NUM_LEADS+1)-1:0] cnt_lead);

    localparam N       = 4;
    localparam idle    = 4'b0000;
    localparam cal_inteval = 4'b0001;
    localparam save_fea = 4'b0111;
    localparam init_ave = 4'b0011;  // use current beat
    localparam update_ave   = 4'b1000; // use previous beat
    localparam cal_fea = 4'b1110;
    localparam done = 4'b1100;
    localparam cal_mi_fea = 4'b1010;
    localparam save_mi_fea = 4'b1011;

    reg         [N-1:0]        feature_state_c         ; // current state
    reg         [N-1:0]        feature_state_n         ; // next state

    
    localparam  NUM_FEAS = 8;
    // localparam  NUM_FEAS_MI = 7; // r_amp-t_amp,q_amp-iso_line, s_amp-iso_line,t_amp-iso_line,r_amp-iso_line,st_amp- iso_line,st_slo, qrs_mi, t_dur_mi,
    localparam  NUM_INTEVALS = 2;
    localparam  SAVE_NUM_BEATS =INIT_NUM_BEATS +1 ;

    
    reg [$clog2(NUM_FEAS+1)-1:0] cnt_feas;
    reg [$clog2(INIT_NUM_BEATS+1)-1:0] cnt_beats;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            feature_state_c <= idle;
        else
            feature_state_c <= feature_state_n;
    end    
    always @(*) begin
        case (feature_state_c)
            idle: begin
                if (mode == 0) begin
                    if (feature_rdy)
                        feature_state_n = cal_inteval; //need to change
                    else
                        feature_state_n = idle;
                end
                else begin
                    if (feature_rdy) begin
                        if (cnt_lead == 0)
                            feature_state_n = cal_inteval; 
                        else 
                            feature_state_n = cal_mi_fea; 
                    end
                    else
                        feature_state_n = idle;                    
                end
            end
            cal_inteval: begin
                if (cnt_feas == NUM_INTEVALS) 
                    if (cnt_beats == INIT_NUM_BEATS+1) begin
                        if (predict_pre == 1) feature_state_n = save_fea;
                        else feature_state_n = cal_fea;
                    end
                    else if (cnt_beats == INIT_NUM_BEATS) feature_state_n = cal_fea;
                    else feature_state_n = save_fea;  
                else
                    feature_state_n = cal_inteval;
            end
            save_fea: begin
                if (cnt_beats == INIT_NUM_BEATS+1) feature_state_n = update_ave;
                else feature_state_n = init_ave;                
            end
            init_ave: begin
                if (cnt_feas == NUM_FEAS) begin
                    if (cnt_beats == INIT_NUM_BEATS) feature_state_n = cal_fea;
                    else  begin
                        if (mode == 0) feature_state_n = done;
                        else feature_state_n = cal_mi_fea;
                    end
                    
                end
                else
                    feature_state_n = init_ave;                
            end
            update_ave: begin
                if (cnt_feas == NUM_FEAS+1)
                    feature_state_n = cal_fea;
                else
                    feature_state_n = update_ave;                   
            end
            cal_fea: begin
                if (cnt_feas == NUM_FEAS) begin
                    if (mode == 0) feature_state_n = done;
                    else feature_state_n = cal_mi_fea;
                end
                    
                else
                    feature_state_n = cal_fea;                  
            end
            cal_mi_fea: begin
                if (cnt_feas == NUM_FEAS_MI + 5)
                    feature_state_n = save_mi_fea;
                else
                    feature_state_n = cal_mi_fea;                
            end
            save_mi_fea: begin
                feature_state_n = done;
            end
            done:
            feature_state_n         = idle;
            default:feature_state_n = idle;
        endcase
    end    
    assign feature_done = (feature_state_c == done)? 1:0;
    assign init_features_end = (cnt_beats >= INIT_NUM_BEATS)? 1:0;
    always @(*) begin
        if (feature_state_c == save_fea) begin
            if (cnt_beats == INIT_NUM_BEATS+1) save_fea_en = 2'b11;
            else save_fea_en = 2'b01;
        end
        else if (feature_state_c == save_mi_fea) begin
            save_fea_en = 2'b10;
        end
        else save_fea_en = 0;
    end
    // assign save_fea_en = (feature_state_c == save_fea)? ((cnt_beats == INIT_NUM_BEATS+1)? 2'b11:2'b01):0;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) cnt_feas <= 0;
        else begin
            if (feature_state_c == cal_inteval) cnt_feas <= (cnt_feas == NUM_INTEVALS)? 0 : cnt_feas + 1;
            else if (feature_state_c == init_ave) cnt_feas <= (cnt_feas == NUM_FEAS)? 0 : cnt_feas + 1;
            else if (feature_state_c == update_ave) cnt_feas <= (cnt_feas == NUM_FEAS+1)? 0 : cnt_feas + 1;
            else if (feature_state_c == cal_fea) cnt_feas <= (cnt_feas == NUM_FEAS)? 0 : cnt_feas + 1;
            else if (feature_state_c == cal_mi_fea) cnt_feas <= (cnt_feas == NUM_FEAS_MI + 5) ? 0: cnt_feas + 1;
            else cnt_feas <= 0; 
        end
    end

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) cnt_beats <= 0;
        else begin
            if (mode==0) begin
                if (feature_state_c == done) cnt_beats <= (cnt_beats == INIT_NUM_BEATS+1)? cnt_beats : cnt_beats + 1;
                else cnt_beats <= cnt_beats;
            end
            else begin
                if ((feature_state_c == done)& (cnt_lead == NUM_LEADS-1)) cnt_beats <= (cnt_beats == INIT_NUM_BEATS+1)? cnt_beats : cnt_beats + 1;
                else cnt_beats <= cnt_beats;                
            end
        end
    end    

    //rr_diff, rr_pre_rr_ave, rr_post_rr_ave, qrs_cur_qrs_ave, r_amp_r_amp_ave, q_amp_q_amp_ave, s_amp_s_amp_ave, p_amp_p_ave, t_amp_t_ave, t_dir

    localparam ADDER_DW = FEATURE_SUM_DW;
    localparam SUB_DW = FEATURE_SUM_DW;
    reg signed [ADDER_DW-1:0] addend1;
    reg signed [ADDER_DW-1:0] addend2;
    wire  signed [ADDER_DW-1:0] sum;
    reg signed [SUB_DW-1:0] minuend;
    reg signed [SUB_DW-1:0] subtrahend;
    wire signed [SUB_DW:0] difference;

    assign sum = addend1 + addend2;
    assign difference = minuend - subtrahend;


    reg signed [ADDER_DW -1 : 0] sum_rr_pre;
    reg signed [ADDER_DW -1: 0] sum_qrs;
    reg signed [ADDER_DW -1: 0] sum_r_amp;
    reg signed [ADDER_DW -1: 0] sum_s_amp;
    reg signed [ADDER_DW -1: 0] sum_q_amp;
    reg signed [ADDER_DW -1: 0] sum_p_amp;
    reg signed [ADDER_DW -1: 0] sum_t_amp;

    reg signed [ADDER_DW -1 : 0] sum_rr_pre_temp;
    reg signed [ADDER_DW -1: 0] sum_qrs_temp;
    reg signed [ADDER_DW -1: 0] sum_r_amp_temp;
    reg signed [ADDER_DW -1: 0] sum_s_amp_temp;
    reg signed [ADDER_DW -1: 0] sum_q_amp_temp;
    reg signed [ADDER_DW -1: 0] sum_p_amp_temp;
    reg signed [ADDER_DW -1: 0] sum_t_amp_temp;

    reg signed [INTEVAL_DW -1 : 0] ave_rr_pre;
    reg signed [INTEVAL_DW -1: 0] ave_qrs;
    reg signed [INPUT_DW -1: 0] ave_r_amp;
    reg signed [INPUT_DW -1: 0] ave_s_amp;
    reg signed [INPUT_DW -1: 0] ave_q_amp;
    reg signed [INPUT_DW -1: 0] ave_p_amp;
    reg signed [INPUT_DW -1: 0] ave_t_amp;

    wire signed [INTEVAL_DW -1 : 0] rr_pre_d8;
    wire signed [INTEVAL_DW -1: 0] qrs_d8;
    wire signed [INPUT_DW -1: 0] r_amp_d8;
    wire signed [INPUT_DW -1: 0] s_amp_d8;
    wire signed [INPUT_DW -1: 0] q_amp_d8;
    wire signed [INPUT_DW -1: 0] p_amp_d8;
    wire signed [INPUT_DW -1: 0] t_amp_d8;

    wire signed [INTEVAL_DW -1 : 0] rr_pre_d1;
    wire signed [INTEVAL_DW -1: 0] qrs_d1;
    wire signed [INPUT_DW -1: 0] r_amp_d1;
    wire signed [INPUT_DW -1: 0] s_amp_d1;
    wire signed [INPUT_DW -1: 0] q_amp_d1;
    wire signed [INPUT_DW -1: 0] p_amp_d1;
    wire signed [INPUT_DW -1: 0] t_amp_d1;

    reg [INPUT_DW -1: 0] st_loc_t_loc;

    assign rr_pre_d8 =  feature_rb[INTEVAL_DW-1-:INTEVAL_DW];
    assign qrs_d8 = feature_rb[SAVE_NUM_BEATS * INTEVAL_DW+INTEVAL_DW-1-:INTEVAL_DW] ;
    assign r_amp_d8 = feature_rb[2*SAVE_NUM_BEATS * INTEVAL_DW+INPUT_DW-1-:INPUT_DW] ;
    assign s_amp_d8 = feature_rb[2*SAVE_NUM_BEATS * INTEVAL_DW+SAVE_NUM_BEATS*INPUT_DW+INPUT_DW-1-:INPUT_DW];
    assign q_amp_d8 = feature_rb[2*SAVE_NUM_BEATS * INTEVAL_DW+2*SAVE_NUM_BEATS*INPUT_DW+INPUT_DW-1-:INPUT_DW];
    assign p_amp_d8 = feature_rb[2*SAVE_NUM_BEATS * INTEVAL_DW+3*SAVE_NUM_BEATS*INPUT_DW+INPUT_DW-1-:INPUT_DW];
    assign t_amp_d8 = feature_rb[2*SAVE_NUM_BEATS * INTEVAL_DW+4*SAVE_NUM_BEATS*INPUT_DW+INPUT_DW-1-:INPUT_DW];

    assign rr_pre_d1 =  feature_rb[SAVE_NUM_BEATS*INTEVAL_DW-1-:INTEVAL_DW];
    assign qrs_d1 = feature_rb[SAVE_NUM_BEATS * INTEVAL_DW+SAVE_NUM_BEATS*INTEVAL_DW-1-:INTEVAL_DW] ;
    assign r_amp_d1 = feature_rb[2*SAVE_NUM_BEATS * INTEVAL_DW+SAVE_NUM_BEATS*INPUT_DW-1-:INPUT_DW] ;
    assign s_amp_d1 = feature_rb[2*SAVE_NUM_BEATS * INTEVAL_DW+SAVE_NUM_BEATS*INPUT_DW+SAVE_NUM_BEATS*INPUT_DW-1-:INPUT_DW];
    assign q_amp_d1 = feature_rb[2*SAVE_NUM_BEATS * INTEVAL_DW+2*SAVE_NUM_BEATS*INPUT_DW+SAVE_NUM_BEATS*INPUT_DW-1-:INPUT_DW];
    assign p_amp_d1 = feature_rb[2*SAVE_NUM_BEATS * INTEVAL_DW+3*SAVE_NUM_BEATS*INPUT_DW+SAVE_NUM_BEATS*INPUT_DW-1-:INPUT_DW];
    assign t_amp_d1 = feature_rb[2*SAVE_NUM_BEATS * INTEVAL_DW+4*SAVE_NUM_BEATS*INPUT_DW+SAVE_NUM_BEATS*INPUT_DW-1-:INPUT_DW];

    
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            minuend <= 0;
            subtrahend <= 0;
            rr_diff <= 0;
            qrs <= 0;
            sum_rr_pre_temp <= 0;
            sum_qrs_temp <= 0;
            sum_r_amp_temp <= 0;
            sum_s_amp_temp <= 0;
            sum_q_amp_temp <= 0;
            sum_p_amp_temp <= 0;
            sum_t_amp_temp <= 0;
            rr_pre_rr_ave <= 0;
            rr_post_rr_ave <= 0;
            qrs_cur_qrs_ave <= 0;
            r_amp_r_amp_ave <= 0;
            q_amp_q_amp_ave <= 0;
            s_amp_s_amp_ave <= 0;
            p_amp_p_amp_ave <= 0;
            t_amp_t_amp_ave<= 0;
            // r_amp_t_amp <= 0;
            q_amp_iso <= 0;
            s_amp_iso <= 0;
            t_amp_iso <= 0;
            // r_amp_iso <= 0;
            st_amp_iso <= 0;
            st_slo <= 0;
            st_loc_t_loc <= 0;


        end 
        else begin
            if (feature_state_c == cal_inteval)  begin // rr_diff, qrs
                case (cnt_feas)
                    0: begin
                        minuend <= {{(SUB_DW-INTEVAL_DW){rr_post[INTEVAL_DW-1]}},rr_post};
                        subtrahend <= {{(SUB_DW-INTEVAL_DW){rr_pre[INTEVAL_DW-1]}},rr_pre};                        
                    end
                    1: begin
                        minuend <= {{(SUB_DW-INTEVAL_DW){s_loc[INTEVAL_DW-1]}}, s_loc};
                        subtrahend <= {{(SUB_DW-INTEVAL_DW){q_loc[INTEVAL_DW-1]}}, q_loc}; 
                        rr_diff <= difference;

                    end
                    2: begin
                        minuend <= 0;
                        subtrahend <= 0;    
                        qrs <=    difference;                   
                    end
                    default: ;
                endcase
            end
            else if  (feature_state_c == update_ave)  begin
                case (cnt_feas)
                    0: begin
                        minuend <= sum_rr_pre;
                        subtrahend <= {{(SUB_DW-INTEVAL_DW){rr_pre_d8[INTEVAL_DW-1]}},rr_pre_d8};                        
                    end
                    1: begin
                        minuend <= sum_qrs;
                        subtrahend <= {{(SUB_DW-INTEVAL_DW){qrs_d8[INTEVAL_DW-1]}}, qrs_d8}; 
                        sum_rr_pre_temp <= difference;

                    end
                    2: begin
                        minuend <= sum_r_amp;
                        subtrahend <= {{(SUB_DW-INPUT_DW){r_amp_d8[INPUT_DW-1]}}, r_amp_d8}; 
                        sum_qrs_temp <= difference;                
                    end
                    3: begin
                        minuend <= sum_s_amp;
                        subtrahend <= {{(SUB_DW-INPUT_DW){s_amp_d8[INPUT_DW-1]}}, s_amp_d8}; 
                        sum_r_amp_temp <= difference;                         
                    end
                    4: begin
                        minuend <= sum_q_amp;
                        subtrahend <= {{(SUB_DW-INPUT_DW){q_amp_d8[INPUT_DW-1]}}, q_amp_d8}; 
                        sum_s_amp_temp <= difference;                         
                    end
                    5: begin
                        minuend <= sum_p_amp;
                        subtrahend <= {{(SUB_DW-INPUT_DW){p_amp_d8[INPUT_DW-1]}}, p_amp_d8}; 
                        sum_q_amp_temp <= difference;                         
                    end
                    6: begin
                        minuend <= sum_t_amp;
                        subtrahend <= {{(SUB_DW-INPUT_DW){t_amp_d8[INPUT_DW-1]}}, t_amp_d8}; 
                        sum_p_amp_temp <= difference;                         
                    end
                    7: begin
                        minuend <= 0;
                        subtrahend <= 0; 
                        sum_t_amp_temp <= difference;                         
                    end
                    default: ;
                endcase                                
            end
            else if (feature_state_c == cal_fea) begin
                case (cnt_feas)
                    0: begin
                        minuend <= {{(SUB_DW-INTEVAL_DW){rr_pre[INTEVAL_DW-1]}},rr_pre};
                        subtrahend <= {{(SUB_DW-INTEVAL_DW){ave_rr_pre[INTEVAL_DW-1]}},ave_rr_pre};                        
                    end
                    1: begin
                        minuend <= {{(SUB_DW-INTEVAL_DW){rr_post[INTEVAL_DW-1]}}, rr_post};
                        subtrahend <= {{(SUB_DW-INTEVAL_DW){ave_rr_pre[INTEVAL_DW-1]}}, ave_rr_pre}; 
                        rr_pre_rr_ave <= difference;

                    end
                    2: begin
                        minuend <=  {{(SUB_DW-INTEVAL_DW){qrs[INTEVAL_DW-1]}}, qrs};
                        subtrahend <= {{(SUB_DW-INTEVAL_DW){ave_qrs[INTEVAL_DW-1]}}, ave_qrs}; 
                        rr_post_rr_ave <= difference;                
                    end
                    3: begin
                        minuend <= {{(SUB_DW-INPUT_DW){r_amp[INPUT_DW-1]}}, r_amp};
                        subtrahend <= {{(SUB_DW-INPUT_DW){ave_r_amp[INPUT_DW-1]}}, ave_r_amp}; 
                        qrs_cur_qrs_ave <= difference;                        
                    end
                    4: begin
                        minuend <= {{(SUB_DW-INPUT_DW){s_amp[INPUT_DW-1]}}, s_amp};
                        subtrahend <= {{(SUB_DW-INPUT_DW){ave_s_amp[INPUT_DW-1]}}, ave_s_amp}; 
                        r_amp_r_amp_ave <= difference;                         
                    end
                    5: begin
                        minuend <= {{(SUB_DW-INPUT_DW){q_amp[INPUT_DW-1]}}, q_amp};
                        subtrahend <= {{(SUB_DW-INPUT_DW){ave_q_amp[INPUT_DW-1]}}, ave_q_amp}; 
                        q_amp_q_amp_ave <= difference;                         
                    end
                    6: begin
                        minuend <= {{(SUB_DW-INPUT_DW){p_amp[INPUT_DW-1]}}, p_amp};
                        subtrahend <= {{(SUB_DW-INPUT_DW){ave_p_amp[INPUT_DW-1]}}, ave_p_amp}; 
                        s_amp_s_amp_ave <= difference;                         
                    end
                    7: begin
                        minuend <= {{(SUB_DW-INPUT_DW){t_amp[INPUT_DW-1]}}, t_amp}; 
                        subtrahend <= {{(SUB_DW-INPUT_DW){ave_t_amp[INPUT_DW-1]}}, ave_t_amp}; 
                        p_amp_p_amp_ave <= difference;                         
                    end
                    8: begin
                        minuend <= 0;
                        subtrahend <= 0; 
                        t_amp_t_amp_ave <= difference;                         
                    end
                    default: ;
                endcase                 
            end
            else if (feature_state_c == cal_mi_fea) begin
                case (cnt_feas)
                    // 0: begin
                    //     minuend <= {{(SUB_DW-INPUT_DW){r_amp[INPUT_DW-1]}},r_amp};
                    //     subtrahend <= {{(SUB_DW-INPUT_DW){t_amp[INPUT_DW-1]}},t_amp};                         
                    // end
                    0: begin
                        minuend <= {{(SUB_DW-INPUT_DW){q_amp[INPUT_DW-1]}},q_amp};
                        subtrahend <= {{(SUB_DW-INPUT_DW){iso_line[INPUT_DW-1]}},iso_line}; 
                        // r_amp_t_amp <= difference;  
                    end
                    1: begin
                        minuend <= {{(SUB_DW-INPUT_DW){s_amp[INPUT_DW-1]}},s_amp};
                        subtrahend <= {{(SUB_DW-INPUT_DW){iso_line[INPUT_DW-1]}},iso_line}; 
                        q_amp_iso <= difference;
                    end
                    2: begin
                        minuend <= {{(SUB_DW-INPUT_DW){t_amp[INPUT_DW-1]}},t_amp};
                        subtrahend <= {{(SUB_DW-INPUT_DW){iso_line[INPUT_DW-1]}},iso_line}; 
                        s_amp_iso <= difference;
                    end
                    // 3: begin
                    //     minuend <= {{(SUB_DW-INPUT_DW){r_amp[INPUT_DW-1]}},r_amp};
                    //     subtrahend <= {{(SUB_DW-INPUT_DW){iso_line[INPUT_DW-1]}},iso_line}; 
                    //     t_amp_iso <= difference;
                    // end
                    3: begin
                        minuend <= {{(SUB_DW-INPUT_DW){st_amp_2[INPUT_DW-1]}},st_amp_2};
                        subtrahend <= {{(SUB_DW-INPUT_DW){iso_line[INPUT_DW-1]}},iso_line}; 
                        t_amp_iso <= difference;
                    end 
                    4: begin
                        st_amp_iso <= difference;
                        if ((t_loc != LENGTH_IN -1 ) & (r_loc != LENGTH_IN-1)) begin
                            minuend <= {{(SUB_DW-INTEVAL_DW){st_loc[INTEVAL_DW-1]}},st_loc};
                            subtrahend <= {{(SUB_DW-INTEVAL_DW){t_loc[INTEVAL_DW-1]}},t_loc};                             
                        end
                        else begin
                            st_slo <= 0;
                        end   
                    end    
                    5: begin
                        if ((t_loc != LENGTH_IN -1 ) & (r_loc != LENGTH_IN-1)) begin
                            minuend <= 0;
                            subtrahend <= 0;
                            st_loc_t_loc <= difference;                             
                        end
                        else ;
                        
                    end 
                    6: begin
                        if ((t_loc != LENGTH_IN -1 ) & (r_loc != LENGTH_IN-1)) begin
                            if (st_loc_t_loc >=8) begin
                                minuend <= {{(SUB_DW-INPUT_DW){st_amp_6[INPUT_DW-1]}},st_amp_6};
                                subtrahend <= {{(SUB_DW-INPUT_DW){st_amp_2[INPUT_DW-1]}},st_amp_2};                                 
                            end
                            else if (st_loc_t_loc >=4) begin
                                minuend <= {{(SUB_DW-INPUT_DW){st_amp_4[INPUT_DW-1]}},st_amp_4};
                                subtrahend <= {{(SUB_DW-INPUT_DW){st_amp_2[INPUT_DW-1]}},st_amp_2};                                 
                            end
                            else begin
                                minuend <= {{(SUB_DW-INPUT_DW){st_amp_2[INPUT_DW-1]}},st_amp_2};
                                subtrahend <= {{(SUB_DW-INPUT_DW){st_amp[INPUT_DW-1]}},st_amp};                               
                            end
                                                    
                        end
                        else;
                    end
                    7:begin
                        minuend <= 0;
                        subtrahend <= 0;
                        if ((t_loc != LENGTH_IN -1 ) & (r_loc != LENGTH_IN-1)) begin
                            if (st_loc_t_loc >=8) begin
                                st_slo <= difference>>>2;                                  
                            end
                            else if (st_loc_t_loc >=4) begin
                                st_slo <= difference>>>1;                                        
                            end
                            else begin     
                                st_slo <= difference;                          
                            end
                                
                        end
                        else;                    
                    end
                    default: ;                        
                                 
                endcase
            end
        end
    end



    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            addend1 <= 0;
            addend2 <= 0;
            sum_rr_pre <= 0;
            sum_qrs <= 0;
            sum_r_amp <= 0;
            sum_s_amp <= 0;
            sum_q_amp <= 0;
            sum_p_amp <= 0;
            sum_t_amp <= 0;
            // r_amp_t_amp_sum <= 0;
            q_amp_iso_sum <= 0;
            s_amp_iso_sum <= 0;
            t_amp_iso_sum <= 0;
            // r_amp_iso_sum <= 0;
            st_amp_iso_sum <= 0;
            st_slo_sum <= 0;


        end 
        else begin
            if (feature_state_c == init_ave)  begin // rr_diff, qrs
                case (cnt_feas)
                    0: begin
                        addend1 <= sum_rr_pre;
                        addend2 <= {{(ADDER_DW-INTEVAL_DW){rr_pre[INTEVAL_DW-1]}},rr_pre};                        
                    end
                    1: begin
                        addend1 <= sum_qrs;
                        addend2 <= {{(ADDER_DW-INTEVAL_DW){qrs[INTEVAL_DW-1]}}, qrs}; 
                        sum_rr_pre <= sum;

                    end
                    2: begin
                        addend1 <= sum_r_amp;
                        addend2 <= {{(ADDER_DW-INPUT_DW){r_amp[INPUT_DW-1]}}, r_amp}; 
                        sum_qrs <= sum;                
                    end
                    3: begin
                        addend1 <= sum_s_amp;
                        addend2 <= {{(ADDER_DW-INPUT_DW){s_amp[INPUT_DW-1]}}, s_amp}; 
                        sum_r_amp <= sum;                         
                    end
                    4: begin
                        addend1 <= sum_q_amp;
                        addend2 <= {{(ADDER_DW-INPUT_DW){q_amp[INPUT_DW-1]}}, q_amp}; 
                        sum_s_amp <= sum;                         
                    end
                    5: begin
                        addend1 <= sum_p_amp;
                        addend2 <= {{(ADDER_DW-INPUT_DW){p_amp[INPUT_DW-1]}}, p_amp}; 
                        sum_q_amp <= sum;                         
                    end
                    6: begin
                        addend1 <= sum_t_amp;
                        addend2 <= {{(ADDER_DW-INPUT_DW){t_amp[INPUT_DW-1]}}, t_amp}; 
                        sum_p_amp <= sum;                         
                    end
                    7: begin
                        addend1 <= 0;
                        addend2 <= 0; 
                        sum_t_amp <= sum;                         
                    end
                    default: ;
                endcase
            end
            else if  (feature_state_c == update_ave)  begin
                case (cnt_feas)
                    0: begin
                        addend1 <= 0;
                        addend2 <= 0;                         
                    end
                    1: begin
                        addend1 <= 0;
                        addend2 <= 0;                          
                    end
                    2: begin
                        addend1 <= sum_rr_pre_temp;
                        addend2 <= {{(ADDER_DW-INTEVAL_DW){rr_pre_d1[INTEVAL_DW-1]}},rr_pre_d1};                        
                    end
                    3: begin
                        addend1 <= sum_qrs_temp;
                        addend2 <= {{(ADDER_DW-INTEVAL_DW){qrs_d1[INTEVAL_DW-1]}}, qrs_d1}; 
                        sum_rr_pre <= sum;

                    end
                    4: begin
                        addend1 <= sum_r_amp_temp;
                        addend2 <= {{(ADDER_DW-INPUT_DW){r_amp_d1[INPUT_DW-1]}}, r_amp_d1}; 
                        sum_qrs <= sum;                
                    end
                    5: begin
                        addend1 <= sum_s_amp_temp;
                        addend2 <= {{(ADDER_DW-INPUT_DW){s_amp_d1[INPUT_DW-1]}}, s_amp_d1}; 
                        sum_r_amp <= sum;                         
                    end
                    6: begin
                        addend1 <= sum_q_amp_temp;
                        addend2 <= {{(ADDER_DW-INPUT_DW){q_amp_d1[INPUT_DW-1]}}, q_amp_d1}; 
                        sum_s_amp <= sum;                         
                    end
                    7: begin
                        addend1 <= sum_p_amp_temp;
                        addend2 <= {{(ADDER_DW-INPUT_DW){p_amp_d1[INPUT_DW-1]}}, p_amp_d1}; 
                        sum_q_amp <= sum;                         
                    end
                    8: begin
                        addend1 <= sum_t_amp_temp;
                        addend2 <= {{(ADDER_DW-INPUT_DW){t_amp_d1[INPUT_DW-1]}}, t_amp_d1}; 
                        sum_p_amp <= sum;                         
                    end
                    9: begin
                        addend1 <= 0;
                        addend2 <= 0; 
                        sum_t_amp <= sum;                         
                    end
                    default: ;
                endcase                
            end
            else if (feature_state_c == cal_mi_fea) begin
                case (cnt_feas)
                0:begin
                    if (cnt_lead == 0) begin //rst
                        // r_amp_t_amp_sum <= 0;
                        q_amp_iso_sum <= 0;
                        s_amp_iso_sum <= 0;
                        t_amp_iso_sum <= 0;
                        // r_amp_iso_sum <= 0;
                        st_amp_iso_sum <= 0;
                        st_slo_sum <= 0;
                    end
                    else begin
                        // r_amp_t_amp_sum <= r_amp_t_amp_sum;
                        q_amp_iso_sum <= q_amp_iso_sum;
                        s_amp_iso_sum <= s_amp_iso_sum;
                        t_amp_iso_sum <= t_amp_iso_sum;
                        // r_amp_iso_sum <= r_amp_iso_sum;
                        st_amp_iso_sum <= st_amp_iso_sum;
                        st_slo_sum <= st_slo_sum;                        
                    end
                end
                // 2:begin
                //     addend1 <= r_amp_t_amp_sum;
                //     addend2 <= {{(ADDER_DW-INPUT_DW){r_amp_t_amp[INPUT_DW-1]}}, r_amp_t_amp};                     
                // end
                2:begin
                    // r_amp_t_amp_sum <= sum;
                    addend1 <= q_amp_iso_sum;
                    addend2 <= {{(ADDER_DW-INPUT_DW){q_amp_iso[INPUT_DW-1]}}, q_amp_iso};                     
                end
                3:begin
                    q_amp_iso_sum <= sum;
                    addend1 <= s_amp_iso_sum;
                    addend2 <= {{(ADDER_DW-INPUT_DW){s_amp_iso[INPUT_DW-1]}}, s_amp_iso};                      
                end
                4:begin
                    s_amp_iso_sum <= sum;
                    addend1 <= t_amp_iso_sum;
                    addend2 <= {{(ADDER_DW-INPUT_DW){t_amp_iso[INPUT_DW-1]}}, t_amp_iso};                      
                end
                // 6:begin
                //     t_amp_iso_sum <= sum;
                //     addend1 <= r_amp_iso_sum;
                //     addend2 <= {{(ADDER_DW-INPUT_DW){r_amp_iso[INPUT_DW-1]}}, r_amp_iso};                      
                // end
                5:begin
                    t_amp_iso_sum <= sum;
                    addend1 <= st_amp_iso_sum;
                    addend2 <= {{(ADDER_DW-INPUT_DW){st_amp_iso[INPUT_DW-1]}}, st_amp_iso};                      
                end
                6:begin
                    st_amp_iso_sum  <=  sum;
                    addend1 <= 0;
                    addend2 <= 0;                      
                end
                7:begin
                    addend1 <= 0;
                    addend2 <= 0;                      
                end
                8:begin
                    addend1 <= st_slo_sum;
                    addend2 <= {{(ADDER_DW-INPUT_DW){st_slo[INPUT_DW-1]}}, st_slo};                     
                end
                9:begin
                    addend1 <= 0;
                    addend2 <= 0;  
                    st_slo_sum  <=  sum;                   
                end
                endcase
            end
            
        end
    end
    always @(*) begin
        ave_rr_pre = sum_rr_pre>>>3;
        ave_qrs = sum_qrs >>> 3;
        ave_r_amp = sum_r_amp >>> 3;
        ave_s_amp  = sum_s_amp >>> 3;
        ave_q_amp = sum_q_amp >>> 3;
        ave_p_amp = sum_p_amp >>> 3;
        ave_t_amp = sum_t_amp >>> 3;        
    end
endmodule
