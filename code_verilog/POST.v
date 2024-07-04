`timescale  1ns/100ps
module POST #(parameter INPUT_DW = 12,
    DATA_DW = 8,
    LENGTH_IN = 256,
    INTEVAL_DW  = $clog2(LENGTH_IN+1),
    NUM_WAVE = 4,
    LABEL_DW = 2,
    TREND_DW = 4,
    DIR_DW = 2,
    EMB_DW = 2,
    QRS_EMB_LEN = 24,
    T_EMB_LEN = 29,
    BG_MIN_LEN = 15,
    PQRST_MIN_LEN = 3,
    ACTIVATION_BUF_LEN1 = 32*64,
    ACTIVATION_BUF_LEN2 = 32*64
)(
    input wclk,
    input rst_n,   
    input post_rdy,
    input [ACTIVATION_BUF_LEN1-1:0] act_sr1,
    input [ACTIVATION_BUF_LEN2-1:0] act_sr2,
    output [3:0] post_state,
    output [4:0] refine_state,
    output reg [$clog2(LENGTH_IN+1)-1:0] wave_duration,
    output reg modify_en,
    output connection_shift,
    output refine_shift_re,
    output refine_shift,
    output emb_shift,

    output reg [$clog2(LENGTH_IN+1)-1:0] r_loc,
    output  signed [INPUT_DW-1:0] r_amp_final,
    output reg [$clog2(LENGTH_IN+1)-1:0] t_on_loc,
    output reg signed [INPUT_DW-1:0] t_on_amp,
    output reg [$clog2(LENGTH_IN+1)-1:0] t_off_loc,
    output reg signed [INPUT_DW-1:0] t_off_amp,
    output reg [$clog2(LENGTH_IN+1)-1:0] t_loc,
    output  signed [INPUT_DW-1:0] t_amp_final,
    output reg signed [DIR_DW-1:0] t_dir,
    output reg [$clog2(LENGTH_IN+1)-1:0] p_on_loc,
    output reg signed [INPUT_DW-1:0] p_on_amp,
    output reg [$clog2(LENGTH_IN+1)-1:0] p_off_loc,
    output reg signed [INPUT_DW-1:0] p_off_amp,
    output reg [$clog2(LENGTH_IN+1)-1:0] p_loc,
    output  signed [INPUT_DW-1:0] p_amp_final,
    output reg signed [DIR_DW-1:0] p_dir,
    output reg [$clog2(LENGTH_IN+1)-1:0] q_loc,
    output  signed [INPUT_DW-1:0] q_amp_final,
    output reg [$clog2(LENGTH_IN+1)-1:0] pq_loc,
    output reg signed [INPUT_DW-1:0] pq_amp,
    output reg [$clog2(LENGTH_IN+1)-1:0] s_loc,
    output  signed [INPUT_DW-1:0] s_amp_final,
    output reg [$clog2(LENGTH_IN+1)-1:0] st_loc,
    output reg signed [INPUT_DW-1:0] st_amp,
    output post_done,
    input mode,
    input feature_done, // rst control signal
    input ann_done,
    output reg signed [INPUT_DW-1:0] st_amp_1,
    output reg signed [INPUT_DW-1:0] st_amp_2,
    output reg signed [INPUT_DW-1:0] st_amp_4,
    output reg signed [INPUT_DW-1:0] st_amp_6,
    output reg signed [INPUT_DW-1:0] iso_line,
    output reg [EMB_DW*QRS_EMB_LEN-1:0] qrs_emb_buffer,
    output reg [EMB_DW*T_EMB_LEN-1:0] t_emb_buffer);


    reg signed [INPUT_DW-1:0] r_amp;
    reg signed [INPUT_DW-1:0] t_amp;
    reg signed [INPUT_DW-1:0] p_amp;
    reg signed [INPUT_DW-1:0] q_amp;
    reg signed [INPUT_DW-1:0] s_amp;

    assign  r_amp_final = (r_amp == {1'B1,{(INPUT_DW-1){1'b0}}})? 0: r_amp;
    assign  t_amp_final = (t_amp == {1'B1,{(INPUT_DW-1){1'b0}}})? 0: t_amp;
    assign  p_amp_final = (p_amp == {1'B1,{(INPUT_DW-1){1'b0}}})? 0: p_amp;
    assign  q_amp_final = (q_amp == {1'B1,{(INPUT_DW-1){1'b0}}})? 0: q_amp;
    assign  s_amp_final = (s_amp ==  {1'B1,{(INPUT_DW-1){1'b0}}})? 0: s_amp;
    
    // assign  r_amp_final =  r_amp;
    // assign  t_amp_final =  t_amp;
    // assign  p_amp_final =  p_amp;
    // assign  q_amp_final =  q_amp;
    // assign  s_amp_final =  s_amp;

    localparam N       = 4;
    localparam idle    = 4'b0000;
    localparam prepare_glb = 4'b0101;
    localparam connection = 4'b0001;
    localparam refine = 4'b0011;
    localparam embedding = 4'b0111;    
    localparam done    = 4'b1000;
    
    reg         [N-1:0]        post_state_c         ; // current state
    reg         [N-1:0]        post_state_n         ; // next state

    reg [$clog2(LENGTH_IN+1)-1:0] cnt_check;

    assign post_state = post_state_c;
    wire refine_done;
    wire connection_done;

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            post_state_c <= idle;
        else
            post_state_c <= post_state_n;
    end

    always @(*) begin
        case (post_state_c)
            idle: begin
                if (post_rdy)
                    post_state_n = prepare_glb; //need to change
                else
                    post_state_n = idle;
            end
            prepare_glb : begin
                post_state_n = connection;
            end
            connection: begin
                if (connection_done)
                    post_state_n = refine;
                else
                    post_state_n = connection;
            end
            refine: begin
                
                if (refine_done) begin
                    if (mode == 0) 
                        post_state_n = done;
                    else
                        post_state_n = embedding;
                end
                else
                    post_state_n = refine;                
            end
            embedding: begin
                if (cnt_check == LENGTH_IN -1) post_state_n = done;
                else  post_state_n = embedding;
            end
            done:
            post_state_n         = idle;
            default:post_state_n = idle;
        endcase
    end

    assign post_done = (post_state_c == done)? 1:0;
     
// connection
    localparam connection_idle    = 3'b000;
    localparam check = 3'b001;
    localparam check_rst = 3'b011;
    localparam connection_finish    = 3'b111;
    reg         [3-1:0]        connection_state_c         ; // current state
    reg         [3-1:0]        connection_state_n         ; // next state 
    
    wire [$clog2(LENGTH_IN+1)-1:0] cnt_check_com;
    reg [$clog2(NUM_WAVE+1)-1:0] cnt_wave;

    
    assign connection_done = (connection_state_c == connection_finish)? 1:0;

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            connection_state_c <= connection_idle;
        else
            connection_state_c <= connection_state_n;
    end
    always @(*) begin
        case (connection_state_c)
            connection_idle: begin
                if (post_rdy)
                    connection_state_n = check; //need to change
                else
                    connection_state_n = connection_idle;
            end
            check: begin
                if (cnt_check == LENGTH_IN-1)
                    connection_state_n = check_rst;
                else
                    connection_state_n = check;
            end
            check_rst: begin
                if (cnt_wave == NUM_WAVE-1)
                    connection_state_n = connection_finish;
                else
                    connection_state_n = check;                
            end 
            connection_finish : connection_state_n = connection_idle;
            default:connection_state_n = connection_idle;
        endcase
    end        

// refine
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

    localparam mi_points = 5'd20; // iso_line, st_amp_1, st_amp_2,  st_amp_4, st_amp_6


    localparam up_down_th_r = 4;
    localparam up_down_th_2 = 2;
    localparam up_down_th_1 = 1;
    localparam up_down_th_0 = 0 ;
    localparam up_down_th_10 = 10 ;
    localparam up_down_th_8 = 8 ;
    reg         [5-1:0]        refine_state_c         ; // current state
    reg         [5-1:0]        refine_state_n         ; // next state 

    wire refine_rdy;
    assign refine_rdy = connection_done;

    assign refine_done = (refine_state_c == refine_finish)? 1:0;
    assign  refine_state = refine_state_c;
    localparam NUM_ON_OFF_MAX = 10;
    reg [$clog2(NUM_ON_OFF_MAX+1)-1:0] qrs_info_ptr_wrt;
    reg [$clog2(NUM_ON_OFF_MAX+1)-1:0] p_info_ptr_wrt;
    reg [$clog2(NUM_ON_OFF_MAX+1)-1:0] t_info_ptr_wrt;
    reg [$clog2(NUM_ON_OFF_MAX+1)-1:0] info_ptr_rd;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            refine_state_c <= refine_idle;
        else
            refine_state_c <= refine_state_n;
    end
    always @(*) begin
        case (refine_state_c)
            refine_idle: begin
                if (refine_rdy)
                    refine_state_n = prepare;
                else
                    refine_state_n = refine_idle;
            end
            prepare: begin
                if (cnt_check == LENGTH_IN-1)
                    refine_state_n = select_rough_qs;
                else
                    refine_state_n = prepare;
            end
            select_rough_qs: begin 
                if (info_ptr_rd == qrs_info_ptr_wrt)  begin 
                    if (info_ptr_rd == 0) refine_state_n = refine_finish;
                    else refine_state_n = determine_r;
                end
                else
                    refine_state_n = select_rough_qs;                
            end 
            determine_r: begin
                if (cnt_check == LENGTH_IN-1) 
                    refine_state_n = select_rough_t;
                else
                    refine_state_n = determine_r;                   
            end
            select_rough_t: begin
                if (info_ptr_rd == t_info_ptr_wrt) begin
                    if (info_ptr_rd == 0) refine_state_n = select_rough_p;
                    else refine_state_n = determine_t;
                end
                else
                    refine_state_n = select_rough_t;                 
            end
            determine_t: begin
                if (cnt_check == LENGTH_IN-1) begin
                    refine_state_n = determine_t_wait; 
                end
                else
                    refine_state_n = determine_t;                   
            end
            determine_t_wait:begin
                if (t_loc == LENGTH_IN -1) refine_state_n = select_rough_p;
                else refine_state_n = determine_t_off;                 
            end
            determine_t_off: begin
                if (cnt_check == LENGTH_IN-1) 
                    refine_state_n = determine_t_on;
                else
                    refine_state_n = determine_t_off;                 
            end
            determine_t_on: begin
                if (cnt_check == LENGTH_IN-1) 
                    refine_state_n = select_rough_p;
                else
                    refine_state_n = determine_t_on;                 
            end
            select_rough_p: begin
                if (info_ptr_rd == p_info_ptr_wrt) begin
                    if (info_ptr_rd == 0) refine_state_n = determine_q_re;
                    else refine_state_n = determine_p;
                end
                else
                    refine_state_n = select_rough_p;                 
            end
            determine_p: begin
                if (cnt_check == LENGTH_IN-1) begin
                    refine_state_n = determine_p_wait;          
                end
                else
                    refine_state_n = determine_p;                   
            end
            determine_p_wait:begin
                if (p_loc == LENGTH_IN -1) refine_state_n = determine_q_re;
                else refine_state_n = determine_p_off;                  
            end
            determine_p_off: begin
                if (cnt_check == LENGTH_IN-1) 
                    refine_state_n = determine_p_on;
                else
                    refine_state_n = determine_p_off;                 
            end
            determine_p_on: begin
                if (cnt_check == LENGTH_IN-1) begin
                    if  (p_off_loc == LENGTH_IN -1) refine_state_n = determine_q_re;
                    else refine_state_n = determine_q;
                end
                else
                    refine_state_n = determine_p_on;                 
            end
            determine_q: begin
                if (cnt_check == LENGTH_IN-1) 
                    refine_state_n = determine_q_re;
                else
                    refine_state_n = determine_q;                 
            end
            determine_q_re: begin
                if (cnt_check == LENGTH_IN-1) begin
                    if (t_on_loc == LENGTH_IN -1) refine_state_n = determine_s;
                    else refine_state_n = determine_s_re;
                end
                else
                    refine_state_n = determine_q_re;                  
            end
            determine_s_re: begin
                if (cnt_check == LENGTH_IN-1)
                    refine_state_n = determine_s;
                else
                    refine_state_n = determine_s_re;                   
            end
            determine_s: begin
                if (cnt_check == LENGTH_IN-1)
                    refine_state_n = s_st_modify;
                else
                    refine_state_n = determine_s;                 
            end
            s_st_modify: begin
                if (mode == 0)
                    refine_state_n = refine_finish;  
                else begin
                    refine_state_n = mi_points;
                end                 
            end
            mi_points: begin
                if (cnt_check == LENGTH_IN-1)
                    refine_state_n = refine_finish; 
                else  refine_state_n = mi_points; 
            end
            refine_finish : refine_state_n = refine_idle;
            default: refine_state_n = refine_idle;
        endcase
    end 
   

    always@(posedge wclk or negedge rst_n) begin
        if (!rst_n) cnt_check <= 0;
        else begin
            if (post_state_c == connection)  begin
                if (connection_state_c == check) begin

                    cnt_check <= (cnt_check == LENGTH_IN-1)? 0 : cnt_check + 1;
                end
                else cnt_check <= 0;
            end
            else if (post_state_c == refine) begin
                if ((refine_state_c == prepare)| (refine_state_c == determine_r) | (refine_state_c == determine_t)| (refine_state_c == determine_t_off)
                    | (refine_state_c == determine_t_on) | (refine_state_c == determine_p) | (refine_state_c == determine_p_off)| (refine_state_c == determine_p_on)
                    | (refine_state_c == determine_q) | (refine_state_c == determine_q_re) | (refine_state_c == determine_s) | (refine_state_c == determine_s_re) | (refine_state_c == mi_points)) begin
                     cnt_check <= (cnt_check == LENGTH_IN-1)? 0 : cnt_check + 1;
                end
                
                else cnt_check <= 0;
            end
            else if (post_state_c == embedding) begin
                cnt_check <= (cnt_check == LENGTH_IN-1)? 0 : cnt_check + 1;
            end
            else cnt_check <= 0;
        end
    end
    always@(posedge wclk or negedge rst_n) begin
        if (!rst_n) cnt_wave <= 0;
        else begin
            if (connection_state_c == check_rst) begin

                cnt_wave <=  cnt_wave + 1;
            end
            else if (connection_state_c == connection_finish) cnt_wave <= 0;
            else cnt_wave <= cnt_wave;
        end
    end

assign connection_shift = (connection_state_c == check)?1:0;




wire [LABEL_DW-1:0] label_cur;
reg [LABEL_DW-1:0] label_minus_duration;

assign label_cur = (post_state_c!= idle)? act_sr1[LABEL_DW-1:0]:0;
always @(*) begin
    if (connection_state_c == check) begin
        case(wave_duration)
            0: label_minus_duration = act_sr1[LABEL_DW*LENGTH_IN-1-:LABEL_DW];
            1: label_minus_duration = act_sr1[LABEL_DW*(LENGTH_IN-1)-1-:LABEL_DW];
            2: label_minus_duration = act_sr1[LABEL_DW*(LENGTH_IN-2)-1-:LABEL_DW];
            3: label_minus_duration = act_sr1[LABEL_DW*(LENGTH_IN-3)-1-:LABEL_DW];
            4: label_minus_duration = act_sr1[LABEL_DW*(LENGTH_IN-4)-1-:LABEL_DW];
            5: label_minus_duration = act_sr1[LABEL_DW*(LENGTH_IN-5)-1-:LABEL_DW];
            6: label_minus_duration = act_sr1[LABEL_DW*(LENGTH_IN-6)-1-:LABEL_DW];
            7: label_minus_duration = act_sr1[LABEL_DW*(LENGTH_IN-7)-1-:LABEL_DW];
            8: label_minus_duration = act_sr1[LABEL_DW*(LENGTH_IN-8)-1-:LABEL_DW];
            9: label_minus_duration = act_sr1[LABEL_DW*(LENGTH_IN-9)-1-:LABEL_DW];
            10: label_minus_duration = act_sr1[LABEL_DW*(LENGTH_IN-10)-1-:LABEL_DW];
            11: label_minus_duration = act_sr1[LABEL_DW*(LENGTH_IN-11)-1-:LABEL_DW];
            12: label_minus_duration = act_sr1[LABEL_DW*(LENGTH_IN-12)-1-:LABEL_DW];
            13: label_minus_duration = act_sr1[LABEL_DW*(LENGTH_IN-13)-1-:LABEL_DW];
            14: label_minus_duration = act_sr1[LABEL_DW*(LENGTH_IN-14)-1-:LABEL_DW];
            15: label_minus_duration = act_sr1[LABEL_DW*(LENGTH_IN-15)-1-:LABEL_DW];
            default: label_minus_duration = act_sr1[LABEL_DW*LENGTH_IN-1-:LABEL_DW];
        endcase
    end
    else label_minus_duration = 0;
end

always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        wave_duration <= 0;
    end 
    else begin
        if ((connection_state_c == check) & (cnt_wave == 0)) begin
            if (label_cur == 2'B00) begin
                wave_duration <= wave_duration + 1;
            end
            else begin
                wave_duration <= 0;
            end
        end
        else if ((connection_state_c == check) & (cnt_wave == 1)) begin
            if (label_cur == 2'B01) begin
                wave_duration <= wave_duration + 1;
            end
            else begin
                wave_duration <= 0;
            end
        end
        else if ((connection_state_c == check) & (cnt_wave == 2)) begin
            if (label_cur == 2'B10) begin
                wave_duration <= wave_duration + 1;
            end
            else begin
                wave_duration <= 0;
            end
        end
        else if ((connection_state_c == check) & (cnt_wave == 3)) begin
            if (label_cur == 2'B11) begin
                wave_duration <= wave_duration + 1;
            end
            else begin
                wave_duration <= 0;
            end
        end
        else wave_duration <= 0;
    end
end

always @(*) begin
     if ((connection_state_c == check) & (cnt_wave == 0)) begin
        if ((label_cur != 2'B00) & (wave_duration!=0)) begin
            if (wave_duration <= BG_MIN_LEN) begin
                if (((cnt_check - wave_duration)>0) & (label_cur == label_minus_duration )) begin
                    modify_en = 1;
                end
                else begin
                    modify_en = 0;
                end
            end
            else begin
                modify_en = 0;                
            end
        end
        else begin
            modify_en = 0;            
        end
    end
    else if ((connection_state_c == check) & (cnt_wave == 1)) begin
        if ((label_cur != 2'B01) & (wave_duration!=0)) begin
            if (wave_duration <= PQRST_MIN_LEN) begin
                if (((cnt_check - wave_duration)>0) & (label_cur == label_minus_duration )) begin
                    modify_en = 1;
                end
                else begin
                    modify_en = 0;
                end
            end
            else begin
                modify_en = 0;                
            end
        end
        else begin
            modify_en = 0;            
        end        
    end
    else if ((connection_state_c == check) & (cnt_wave == 2)) begin
        if ((label_cur != 2'B10) & (wave_duration!=0)) begin
            if (wave_duration <= PQRST_MIN_LEN) begin
                if (((cnt_check - wave_duration)>0) & (label_cur == label_minus_duration )) begin
                    modify_en = 1;
                end
                else begin
                    modify_en = 0;
                end
            end
            else begin
                modify_en = 0;                
            end
        end
        else begin
            modify_en = 0;            
        end        
    end   
    else if ((connection_state_c == check) & (cnt_wave == 3)) begin
        if ((label_cur != 2'B11) & (wave_duration!=0)) begin
            if (wave_duration < PQRST_MIN_LEN) begin
                if (((cnt_check - wave_duration)>0) & (label_cur == label_minus_duration )) begin
                    modify_en = 1;
                end
                else begin
                    modify_en = 0;
                end

            end
            else begin
                modify_en = 0;                
            end
        end
        else begin
            modify_en = 0;            
        end        
    end  
    else begin
        modify_en = 0;        
    end
end


reg [$clog2(LENGTH_IN+1)-1:0] qrs_ons [NUM_ON_OFF_MAX-1:0]; // locations of onsets of qrs wave
reg [$clog2(LENGTH_IN+1)-1:0] qrs_offs [NUM_ON_OFF_MAX-1:0];
reg [$clog2(LENGTH_IN+1)-1:0] p_ons [NUM_ON_OFF_MAX-1:0];
reg [$clog2(LENGTH_IN+1)-1:0] p_offs [NUM_ON_OFF_MAX-1:0];
reg [$clog2(LENGTH_IN+1)-1:0] t_ons [NUM_ON_OFF_MAX-1:0];
reg [$clog2(LENGTH_IN+1)-1:0] t_offs [NUM_ON_OFF_MAX-1:0]; 

// reg [$clog2(NUM_ON_OFF_MAX+1)-1:0] p_info_ptr_rd;
// reg [$clog2(NUM_ON_OFF_MAX+1)-1:0] t_info_ptr_rd;

wire [LABEL_DW-1:0] label_pre; // Add one bit for 
assign label_pre = ((refine_state_c == prepare)&(cnt_check >0))? act_sr1[LENGTH_IN* LABEL_DW-1-:LABEL_DW] : 2'b00;
integer i;
integer j;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < NUM_ON_OFF_MAX; i = i+1) begin
            qrs_ons[i] <= 0;
            qrs_offs[i] <= 0;
            p_ons[i] <= 0;
            p_offs[i] <= 0;
            t_ons[i] <= 0;
            t_offs[i] <= 0;
        end
        qrs_info_ptr_wrt <= 0;
        p_info_ptr_wrt <= 0;
        t_info_ptr_wrt <= 0;
    end    
    else begin
        if (refine_state_c == prepare) begin
            case(label_cur)
                2'B01:begin
                    if (label_pre != 2'b01) begin
                        p_ons[p_info_ptr_wrt] <=  cnt_check;
                        p_info_ptr_wrt <= p_info_ptr_wrt ;
                    end
                    else begin
//                        p_ons <= p_ons;
                        p_info_ptr_wrt <= p_info_ptr_wrt ;
                    end
                end
                2'B10: begin
                    if (label_pre != 2'b10) begin
                        qrs_ons[qrs_info_ptr_wrt] <=  cnt_check;
                        qrs_info_ptr_wrt <= qrs_info_ptr_wrt;
                    end
                    else begin
//                        qrs_ons <= qrs_ons;
                        qrs_info_ptr_wrt <= qrs_info_ptr_wrt ;
                    end
                end
                2'B11: begin
                    if (label_pre != 2'b11) begin
                        t_ons[t_info_ptr_wrt] <=  cnt_check;
                        t_info_ptr_wrt <= t_info_ptr_wrt;
                    end
                    else begin
//                        t_ons <= t_ons;
                        t_info_ptr_wrt <= t_info_ptr_wrt ;
                    end                    
                end
                default:;
            endcase
            case (label_pre)
                2'B01:begin
                    if ((label_cur != 2'b01) |(cnt_check == LENGTH_IN -1))begin
                        p_offs[p_info_ptr_wrt] <=  cnt_check;
                        p_info_ptr_wrt <= p_info_ptr_wrt + 1;
                    end
                    else begin
//                        p_offs <= p_offs;
                        p_info_ptr_wrt <= p_info_ptr_wrt ;
                    end
                end
                2'b10: begin
                    if ((label_cur != 2'b10) |(cnt_check == LENGTH_IN -1))begin
                        qrs_offs[qrs_info_ptr_wrt] <=  cnt_check;
                        qrs_info_ptr_wrt <= qrs_info_ptr_wrt  + 1;
                    end
                    else begin
//                        qrs_offs <= qrs_offs;
                        qrs_info_ptr_wrt <= qrs_info_ptr_wrt ;
                    end                    
                end
                2'b11: begin
                    if ((label_cur != 2'b11) |(cnt_check == LENGTH_IN -1))begin
                        t_offs[t_info_ptr_wrt] <=  cnt_check;
                        t_info_ptr_wrt <= t_info_ptr_wrt  + 1;
                    end
                    else begin
//                        t_offs <= t_offs;
                        t_info_ptr_wrt <= t_info_ptr_wrt ;
                    end                     
                end
                default:; 
            endcase
        end
        else if (refine_state_c == refine_finish) begin //reset
            for (j = 0; j < NUM_ON_OFF_MAX-1; j = j+1) begin
                qrs_ons[j] <= 0;
                qrs_offs[j] <= 0;
                p_ons[j] <= 0;
                p_offs[j] <= 0;
                t_ons[j] <= 0;
                t_offs[j] <= 0;
            end
            qrs_info_ptr_wrt <= 0;
            p_info_ptr_wrt <= 0;
            t_info_ptr_wrt <= 0;            
        end
        else;
    end
end
assign refine_shift = ((refine_state_c == prepare) |(refine_state_c == determine_r)|
                (refine_state_c == determine_t)|(refine_state_c == determine_t_off)|
                (refine_state_c ==  determine_p) | (refine_state_c ==  determine_p_off)|
                (refine_state_c ==  determine_s) | (refine_state_c ==  determine_q) |(refine_state_c == mi_points))? 1:0;
assign refine_shift_re = ((refine_state_c == determine_t_on) |(refine_state_c == determine_p_on)|(refine_state_c == determine_s_re) |(refine_state_c == determine_q_re))?1:0;
assign cnt_check_com = ((refine_state_c == determine_t_on) |(refine_state_c == determine_p_on)|(refine_state_c == determine_s_re) |(refine_state_c == determine_q_re))?(LENGTH_IN-cnt_check-1):0;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        info_ptr_rd <= 0;
    end
    else begin
        if (refine_state_c == select_rough_qs) begin
            info_ptr_rd <= (info_ptr_rd == qrs_info_ptr_wrt)? info_ptr_rd:info_ptr_rd+1;
        end
        else if (refine_state_c == determine_r) begin
            info_ptr_rd <= 0; //reset
        end
        else if (refine_state_c == select_rough_t) begin
            info_ptr_rd <= (info_ptr_rd == t_info_ptr_wrt)? info_ptr_rd:info_ptr_rd+1;
        end
        else if (refine_state_c ==  determine_t)  begin
            info_ptr_rd <= 0;
        end
        else if (refine_state_c == select_rough_p) begin
            info_ptr_rd <= (info_ptr_rd == p_info_ptr_wrt)? info_ptr_rd:info_ptr_rd+1;
        end
        else if (refine_state_c ==  determine_p) begin
            info_ptr_rd <= 0;
        end
        else  info_ptr_rd <= info_ptr_rd;
    end
end
reg [$clog2(LENGTH_IN+1)-1:0] on_candidate;
reg [$clog2(LENGTH_IN+1)-1:0] off_candidate;

reg [$clog2(LENGTH_IN+1)-1:0] q_on_candidate_min;
reg [$clog2(LENGTH_IN+1)-1:0] s_off_candidate_min;
reg [$clog2(LENGTH_IN+1)-1:0] q_on_rough;
reg [$clog2(LENGTH_IN+1)-1:0] s_off_rough;
reg find_qs_rough_end;

reg [$clog2(LENGTH_IN+1)-1:0] t_on_rough;
reg [$clog2(LENGTH_IN+1)-1:0] t_off_rough;
reg signed [$clog2(LENGTH_IN+1):0] dis_candidates;
reg signed [$clog2(LENGTH_IN+1):0] dis_candidates_temp;
wire [$clog2(LENGTH_IN+1)-1:0]  t_on_rough_minus1;
wire [$clog2(LENGTH_IN+1)-1:0]  t_off_rough_plus1;
wire [$clog2(LENGTH_IN+1)-1:0]  t_off_rough_16;
wire [$clog2(LENGTH_IN+1)-1:0]  t_on_rough_minus10;

reg [$clog2(LENGTH_IN+1)-1:0] p_on_rough;
reg [$clog2(LENGTH_IN+1)-1:0] p_off_rough;

wire [$clog2(LENGTH_IN+1)-1:0]  p_on_rough_minus1;
wire [$clog2(LENGTH_IN+1)-1:0]  p_off_rough_plus1;
wire [$clog2(LENGTH_IN+1)-1:0]  p_off_rough_16;
wire [$clog2(LENGTH_IN+1)-1:0]  p_on_rough_minus10;

always @(*) begin
    if (refine_state_c == select_rough_qs) begin
        on_candidate = qrs_ons[info_ptr_rd];
        off_candidate = qrs_offs[info_ptr_rd];
    end
    else if (refine_state_c == select_rough_t) begin
        on_candidate = t_ons[info_ptr_rd];
        off_candidate = t_offs[info_ptr_rd];       
    end
    else if (refine_state_c == select_rough_p) begin
        on_candidate = p_ons[info_ptr_rd];
        off_candidate = p_offs[info_ptr_rd];       
    end
    else begin
        on_candidate = 0;
        off_candidate = 0;
    end
end
localparam LENGTH_MID = 106;
reg signed [$clog2(LENGTH_IN+1):0] dis_q_on_candidate_mid;
reg signed [$clog2(LENGTH_IN+1):0] dis_s_off_candidate_mid;
reg signed [$clog2(LENGTH_IN+1):0] dis_qs_candidate_mid;
reg signed [$clog2(LENGTH_IN+1):0] dis_qs_candidate_temp;
always @(*) begin
    if (refine_state_c == select_rough_qs) begin
        dis_q_on_candidate_mid = (on_candidate >LENGTH_MID)? (on_candidate- LENGTH_MID):(LENGTH_MID -on_candidate);
        dis_s_off_candidate_mid = (off_candidate>LENGTH_MID)? (off_candidate- LENGTH_MID):(LENGTH_MID -off_candidate);
        dis_qs_candidate_mid = (dis_q_on_candidate_mid<dis_s_off_candidate_mid)? dis_q_on_candidate_mid:dis_s_off_candidate_mid;
    end
    else begin
        dis_q_on_candidate_mid =0;
        dis_s_off_candidate_mid = 0;
        dis_qs_candidate_mid = 0;        
    end
end
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        q_on_candidate_min <= 0;
        s_off_candidate_min <= 0;
        dis_qs_candidate_temp <= LENGTH_IN-1;
    end
    else begin
        if (refine_state_c == select_rough_qs) begin
            if (dis_qs_candidate_mid<dis_qs_candidate_temp) begin
                q_on_candidate_min <= on_candidate;
                s_off_candidate_min <= off_candidate;
                dis_qs_candidate_temp <= dis_qs_candidate_mid;
            end
            else begin
                q_on_candidate_min <= q_on_candidate_min;
                s_off_candidate_min <= s_off_candidate_min;
                dis_qs_candidate_temp <= dis_qs_candidate_temp;
            end
        end
        else if (refine_state_c == refine_finish) begin
            q_on_candidate_min <= 0;
            s_off_candidate_min <= 0;
            dis_qs_candidate_temp <= LENGTH_IN-1;        
        end
    end
end
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        q_on_rough <= 0;
        s_off_rough <= 0;
        find_qs_rough_end <= 0;
    end
    else begin
        if (refine_state_c == select_rough_qs) begin
            if ((on_candidate <= LENGTH_MID) & (off_candidate>= LENGTH_MID)) begin
                q_on_rough <= on_candidate;
                s_off_rough <= off_candidate;
                find_qs_rough_end <= 1;
            end
            else begin
                if ((!find_qs_rough_end) & (info_ptr_rd == qrs_info_ptr_wrt)) begin // not find
                    q_on_rough <= q_on_candidate_min;
                    s_off_rough <= s_off_candidate_min;
                    find_qs_rough_end <= 1;                    
                end
                else begin
                    q_on_rough <= q_on_rough;
                    s_off_rough  <= s_off_rough; 
                    find_qs_rough_end <=   find_qs_rough_end; 
                end
            end
        end
        else if (refine_state_c == refine_finish) begin
            q_on_rough <= 0;
            s_off_rough <= 0;
            find_qs_rough_end <= 0;             
        end
    end
end


// determine r
wire signed [INPUT_DW-1:0] amp_cur;
wire signed [INPUT_DW-1:0] amp_pre;
wire signed [INPUT_DW-1:0] amp_post;
wire [$clog2(LENGTH_IN+1)-1:0]  q_on_rough_minus1;
wire [$clog2(LENGTH_IN+1)-1:0]  s_off_rough_plus1;
reg [TREND_DW-1:0] trend;
reg [TREND_DW-1:0] trend_d;
reg [TREND_DW-1:0] trend_2d;
reg [TREND_DW-1:0] trend_3d;
reg [TREND_DW-1:0] trend_4d;
reg [TREND_DW-1:0] trend_5d;

wire signed [INPUT_DW:0] amp_pre_minus_cur;
wire signed [INPUT_DW:0] amp_cur_minus_post;
reg flat_pre;
reg up_pre;
reg down_pre;
reg flat_post;
reg up_post;
reg down_post;
wire trend_cal_en;

reg signed [INPUT_DW-1:0] t_pos_temp;
reg [$clog2(LENGTH_IN+1)-1:0] t_pos_loc_temp;
reg signed [INPUT_DW-1:0] t_neg_temp;
reg [$clog2(LENGTH_IN+1)-1:0] t_neg_loc_temp;

assign q_on_rough_minus1 = q_on_rough - 1;
assign s_off_rough_plus1 = s_off_rough + 1;
assign amp_cur = (trend_cal_en| (refine_state_c == mi_points) | (post_state_c == embedding))? act_sr2[INPUT_DW-1:0]:0;
assign amp_post = (trend_cal_en)? act_sr2[2*INPUT_DW-1:INPUT_DW]:0;
assign amp_pre = (trend_cal_en| (post_state_c == embedding))? act_sr2[INPUT_DW*LENGTH_IN-1-:INPUT_DW]:0; 
assign amp_pre_minus_cur = (trend_cal_en|emb_shift)? (amp_pre - amp_cur):0; // need to add enable
assign amp_cur_minus_post = (trend_cal_en)? (amp_cur - amp_post):0;// need to add enable

always @(*) begin // to be optimized
    if (refine_state_c == determine_r) begin
        flat_pre = ((amp_pre_minus_cur<=up_down_th_r)& (amp_pre_minus_cur>=-up_down_th_r))? 1:0;
        up_pre = (amp_pre_minus_cur<-up_down_th_r)? 1:0;
        down_pre =  (amp_pre_minus_cur>up_down_th_r)? 1:0;
        flat_post = ((amp_cur_minus_post<=up_down_th_r)& (amp_cur_minus_post>=-up_down_th_r))? 1:0;
        up_post = (amp_cur_minus_post<-up_down_th_r)? 1:0;
        down_post =  (amp_cur_minus_post>up_down_th_r)? 1:0;        
    end
    else if (refine_state_c == determine_t) begin
            flat_pre = ((amp_pre_minus_cur<=up_down_th_1)& (amp_pre_minus_cur>=-up_down_th_1))? 1:0;
            up_pre = (amp_pre_minus_cur<-up_down_th_1)? 1:0;
            down_pre =  (amp_pre_minus_cur>up_down_th_1)? 1:0;
            flat_post = ((amp_cur_minus_post<=up_down_th_1)& (amp_cur_minus_post>=-up_down_th_1))? 1:0;
            up_post = (amp_cur_minus_post<-up_down_th_1)? 1:0;
            down_post =  (amp_cur_minus_post>up_down_th_1)? 1:0;          
    end
    else if (refine_state_c == determine_t_off) begin
        if (cnt_check < t_off_rough_plus1) begin
            flat_pre = ((amp_pre_minus_cur<=up_down_th_0)& (amp_pre_minus_cur>=-up_down_th_0))? 1:0;
            up_pre = (amp_pre_minus_cur<-up_down_th_0)? 1:0;
            down_pre =  (amp_pre_minus_cur>up_down_th_0)? 1:0;
            flat_post = ((amp_cur_minus_post<=up_down_th_0)& (amp_cur_minus_post>=-up_down_th_0))? 1:0;
            up_post = (amp_cur_minus_post<-up_down_th_0)? 1:0;
            down_post =  (amp_cur_minus_post>up_down_th_0)? 1:0;     
        end 
        else begin
            flat_pre = ((amp_pre_minus_cur<=up_down_th_1)& (amp_pre_minus_cur>=-up_down_th_1))? 1:0;
            up_pre = (amp_pre_minus_cur<-up_down_th_1)? 1:0;
            down_pre =  (amp_pre_minus_cur>up_down_th_1)? 1:0;
            flat_post = ((amp_cur_minus_post<=up_down_th_1)& (amp_cur_minus_post>=-up_down_th_1))? 1:0;
            up_post = (amp_cur_minus_post<-up_down_th_1)? 1:0;
            down_post =  (amp_cur_minus_post>up_down_th_1)? 1:0;            
        end  
    end 
    else if (refine_state_c == determine_t_on) begin
        if (cnt_check_com > t_on_rough_minus1) begin
            flat_pre = ((amp_pre_minus_cur<=up_down_th_0)& (amp_pre_minus_cur>=-up_down_th_0))? 1:0;
            up_pre = (amp_pre_minus_cur<-up_down_th_0)? 1:0;
            down_pre =  (amp_pre_minus_cur>up_down_th_0)? 1:0;
            flat_post = ((amp_cur_minus_post<=up_down_th_0)& (amp_cur_minus_post>=-up_down_th_0))? 1:0;
            up_post = (amp_cur_minus_post<-up_down_th_0)? 1:0;
            down_post =  (amp_cur_minus_post>up_down_th_0)? 1:0;     
        end 
        else begin
            flat_pre = ((amp_pre_minus_cur<=up_down_th_1)& (amp_pre_minus_cur>=-up_down_th_1))? 1:0;
            up_pre = (amp_pre_minus_cur<-up_down_th_1)? 1:0;
            down_pre =  (amp_pre_minus_cur>up_down_th_1)? 1:0;
            flat_post = ((amp_cur_minus_post<=up_down_th_1)& (amp_cur_minus_post>=-up_down_th_1))? 1:0;
            up_post = (amp_cur_minus_post<-up_down_th_1)? 1:0;
            down_post =  (amp_cur_minus_post>up_down_th_1)? 1:0;            
        end  
    end    
    else if (refine_state_c == determine_p) begin
            flat_pre = ((amp_pre_minus_cur<=up_down_th_0)& (amp_pre_minus_cur>=-up_down_th_0))? 1:0;
            up_pre = (amp_pre_minus_cur<-up_down_th_0)? 1:0;
            down_pre =  (amp_pre_minus_cur>up_down_th_0)? 1:0;
            flat_post = ((amp_cur_minus_post<=up_down_th_0)& (amp_cur_minus_post>=-up_down_th_0))? 1:0;
            up_post = (amp_cur_minus_post<-up_down_th_0)? 1:0;
            down_post =  (amp_cur_minus_post>up_down_th_0)? 1:0;          
    end
    else if (refine_state_c == determine_p_off) begin
        if (cnt_check < p_off_rough_plus1) begin
            flat_pre = ((amp_pre_minus_cur<=up_down_th_0)& (amp_pre_minus_cur>=-up_down_th_0))? 1:0;
            up_pre = (amp_pre_minus_cur<-up_down_th_0)? 1:0;
            down_pre =  (amp_pre_minus_cur>up_down_th_0)? 1:0;
            flat_post = ((amp_cur_minus_post<=up_down_th_0)& (amp_cur_minus_post>=-up_down_th_0))? 1:0;
            up_post = (amp_cur_minus_post<-up_down_th_0)? 1:0;
            down_post =  (amp_cur_minus_post>up_down_th_0)? 1:0;     
        end 
        else begin
            flat_pre = ((amp_pre_minus_cur<=up_down_th_2)& (amp_pre_minus_cur>=-up_down_th_2))? 1:0;
            up_pre = (amp_pre_minus_cur<-up_down_th_2)? 1:0;
            down_pre =  (amp_pre_minus_cur>up_down_th_2)? 1:0;
            flat_post = ((amp_cur_minus_post<=up_down_th_2)& (amp_cur_minus_post>=-up_down_th_2))? 1:0;
            up_post = (amp_cur_minus_post<-up_down_th_2)? 1:0;
            down_post =  (amp_cur_minus_post>up_down_th_2)? 1:0;            
        end  
    end
    else if (refine_state_c == determine_p_on) begin
        if (cnt_check_com > p_on_rough_minus1) begin
            flat_pre = ((amp_pre_minus_cur<=up_down_th_0)& (amp_pre_minus_cur>=-up_down_th_0))? 1:0;
            up_pre = (amp_pre_minus_cur<-up_down_th_0)? 1:0;
            down_pre =  (amp_pre_minus_cur>up_down_th_0)? 1:0;
            flat_post = ((amp_cur_minus_post<=up_down_th_0)& (amp_cur_minus_post>=-up_down_th_0))? 1:0;
            up_post = (amp_cur_minus_post<-up_down_th_0)? 1:0;
            down_post =  (amp_cur_minus_post>up_down_th_0)? 1:0;     
        end 
        else begin
            flat_pre = ((amp_pre_minus_cur<=up_down_th_2)& (amp_pre_minus_cur>=-up_down_th_2))? 1:0;
            up_pre = (amp_pre_minus_cur<-up_down_th_2)? 1:0;
            down_pre =  (amp_pre_minus_cur>up_down_th_2)? 1:0;
            flat_post = ((amp_cur_minus_post<=up_down_th_2)& (amp_cur_minus_post>=-up_down_th_2))? 1:0;
            up_post = (amp_cur_minus_post<-up_down_th_2)? 1:0;
            down_post =  (amp_cur_minus_post>up_down_th_2)? 1:0;            
        end  
    end 
    else if ((refine_state_c == determine_q)|(refine_state_c == determine_q_re))  begin
            flat_pre = ((amp_pre_minus_cur<=up_down_th_1)& (amp_pre_minus_cur>=-up_down_th_1))? 1:0;
            up_pre = (amp_pre_minus_cur<-up_down_th_1)? 1:0;
            down_pre =  (amp_pre_minus_cur>up_down_th_1)? 1:0;
            flat_post = ((amp_cur_minus_post<=up_down_th_1)& (amp_cur_minus_post>=-up_down_th_1))? 1:0;
            up_post = (amp_cur_minus_post<-up_down_th_1)? 1:0;
            down_post =  (amp_cur_minus_post>up_down_th_1)? 1:0;        
    end   
    else if ((refine_state_c == determine_s)|(refine_state_c == determine_s_re)) begin
        if (t_on_loc != LENGTH_IN -1) begin
            flat_pre = ((amp_pre_minus_cur<=up_down_th_10)& (amp_pre_minus_cur>=-up_down_th_10))? 1:0;
            up_pre = (amp_pre_minus_cur<-up_down_th_10)? 1:0;
            down_pre =  (amp_pre_minus_cur>up_down_th_10)? 1:0;
            flat_post = ((amp_cur_minus_post<=up_down_th_10)& (amp_cur_minus_post>=-up_down_th_10))? 1:0;
            up_post = (amp_cur_minus_post<-up_down_th_10)? 1:0;
            down_post =  (amp_cur_minus_post>up_down_th_10)? 1:0; 
        end
        else begin
            flat_pre = ((amp_pre_minus_cur<=up_down_th_8)& (amp_pre_minus_cur>=-up_down_th_8))? 1:0;
            up_pre = (amp_pre_minus_cur<-up_down_th_8)? 1:0;
            down_pre =  (amp_pre_minus_cur>up_down_th_8)? 1:0;
            flat_post = ((amp_cur_minus_post<=up_down_th_8)& (amp_cur_minus_post>=-up_down_th_8))? 1:0;
            up_post = (amp_cur_minus_post<-up_down_th_8)? 1:0;
            down_post =  (amp_cur_minus_post>up_down_th_8)? 1:0; 
        end
    end
    else begin
            flat_pre = 0;
            up_pre = 0;
            down_pre =  0;
            flat_post = 0;
            up_post = 0;
            down_post =  0;         
    end
end

assign trend_cal_en = (((refine_state_c == determine_r) & (cnt_check >= q_on_rough_minus1) & (cnt_check <= s_off_rough_plus1))|
                ((refine_state_c == determine_t) & (cnt_check >= t_on_rough_minus1) & (cnt_check <= t_off_rough_plus1))|
                ((refine_state_c == determine_t_off) & (cnt_check >= t_on_rough_minus1) & (cnt_check <= t_off_rough_16))|
                ((refine_state_c == determine_t_on) & (cnt_check_com >t_on_rough_minus10 ) & (cnt_check_com <= t_off_rough_plus1))|
                ((refine_state_c == determine_p) & (cnt_check >= p_on_rough_minus1) & (cnt_check <= p_off_rough_plus1))|
                ((refine_state_c == determine_p_off) & (cnt_check >= p_on_rough_minus1) & (cnt_check <= p_off_rough_16))|
                ((refine_state_c == determine_p_on) & (cnt_check_com> p_on_rough_minus10 ) & (cnt_check_com <= p_off_rough_plus1))|
                ((refine_state_c == determine_q) & (cnt_check >= p_off_loc) & (cnt_check <= r_loc ))|
                ((refine_state_c == determine_q_re) & (cnt_check_com <= r_loc))|
                ((refine_state_c == determine_s) & (cnt_check >= r_loc))|
                ((refine_state_c == determine_s_re) & (cnt_check_com >=r_loc) & (cnt_check_com <= t_on_loc ))
                )? 1:0;
always @(*) begin
    if (trend_cal_en)  begin
        if (flat_pre & flat_post ) trend = 0;
        else if (flat_pre & up_post) trend = 1;
        else if (flat_pre & down_post) trend = 2;
        else if (up_pre & flat_post) trend = 3;
        else if (up_pre & up_post) trend = 4;
        else if (up_pre & down_post) trend = 5;
        else if (down_pre & flat_post) trend = 6;
        else if (down_pre & up_post) trend = 7;
        else if (down_pre & down_post) trend = 8;
        else trend = 9;
    end
    else trend = 9;
end
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        trend_d <= 0;
        trend_2d <= 0;
        trend_3d <= 0;
        trend_4d <= 0;
        trend_5d <= 0;
    end
    else begin
        trend_d <= trend;
        trend_2d <= trend_d;
        trend_3d <= trend_2d;
        trend_4d <= trend_3d;
        trend_5d <= trend_4d;
    end
end

always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        r_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
        r_loc <= LENGTH_IN - 1;

    end
    else begin // lack reset
        if (refine_state_c == determine_r) begin
            if ((cnt_check >= q_on_rough_minus1) & (cnt_check <=  s_off_rough_plus1)) begin
                if ((trend == 5) | ((trend == 2) & (trend_d == 3))|((trend == 2) & (trend_d == 0) & (trend_2d == 3))) begin
                    if (amp_cur > r_amp)  begin
                        r_loc <= cnt_check;
                        r_amp <= amp_cur;
                    end
                    else begin
                        r_loc <= r_loc;
                        r_amp <= r_amp;                        
                    end
                end
            end
            else begin
                r_loc <= r_loc;
                r_amp <= r_amp;   
            end
        end
        else begin

            if (feature_done) begin
                r_loc <= LENGTH_IN - 1;
                r_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};              
            end
            else begin
                r_loc <= r_loc;
                r_amp <= r_amp;   
            end
   
        end
    end
end
//  select rough t/p

always @(*) begin
    if (refine_state_c == select_rough_t) begin
        dis_candidates = (on_candidate > s_off_rough)? (on_candidate- s_off_rough):LENGTH_IN-1;
    end
    else if  (refine_state_c == select_rough_p) begin
        dis_candidates = (q_on_rough > off_candidate)? (q_on_rough - off_candidate):LENGTH_IN-1;
    end
    else dis_candidates = 0;
end
// wire test1;
// wire test2;
// wire test3;
// assign test1 = (dis_candidates<dis_candidates_temp)? 1: 0;
// assign test2 = (dis_candidates < 150)? 1: 0;
// assign test3 = (refine_state_c == select_rough_t)?1:0;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        t_on_rough <=  LENGTH_IN-1;
        t_off_rough <=  LENGTH_IN-1;
        p_on_rough <= LENGTH_IN-1;
        p_off_rough <= LENGTH_IN-1;
        dis_candidates_temp <= LENGTH_IN-1;
    end
    else begin
        if (refine_state_c == select_rough_t) begin
            if (dis_candidates<dis_candidates_temp) begin
                if (dis_candidates < 150) begin
                    t_on_rough <= on_candidate;
                    t_off_rough <= off_candidate;                 
                    dis_candidates_temp <= dis_candidates;
                end
                else begin
                    t_on_rough <= t_on_rough;
                    t_off_rough <= t_off_rough;
                    dis_candidates_temp <= dis_candidates_temp;                    
                end
            end
            else begin
                t_on_rough <= t_on_rough;
                t_off_rough <= t_off_rough;
                dis_candidates_temp <= dis_candidates_temp;
            end
        end
        else if (refine_state_c == determine_t) begin
                t_on_rough <= t_on_rough;
                t_off_rough <= t_off_rough;
                dis_candidates_temp <=  LENGTH_IN-1;            
        end
        else if (refine_state_c == select_rough_p) begin
            if (dis_candidates<dis_candidates_temp) begin
                if (dis_candidates < 150) begin
                    p_on_rough <= on_candidate;
                    p_off_rough <= off_candidate;                 
                    dis_candidates_temp <= dis_candidates;
                end
                else begin
                    p_on_rough <= p_on_rough;
                    p_off_rough <= p_off_rough;
                    dis_candidates_temp <= dis_candidates_temp;                    
                end
            end
            else begin
                p_on_rough <= p_on_rough;
                p_off_rough <= p_off_rough;
                dis_candidates_temp <= dis_candidates_temp;
            end            
        end
        else if (refine_state_c == refine_finish) begin
            t_on_rough <= LENGTH_IN-1;
            t_off_rough <=  LENGTH_IN-1;
            p_on_rough <= LENGTH_IN-1;
            p_off_rough <= LENGTH_IN-1;            
            dis_candidates_temp <= LENGTH_IN-1;        
        end
        else;
    end
end
// determine t


reg signed [INPUT_DW-1:0] off_rough_amp;
reg signed [INPUT_DW-1:0] on_rough_amp;
reg signed [INPUT_DW-1:0] tp_max;
reg [$clog2(LENGTH_IN+1)-1:0]  tp_max_loc;

assign t_on_rough_minus1 = t_on_rough - 1;
assign t_off_rough_plus1 = t_off_rough + 1;
assign t_off_rough_16 = t_off_rough + 16;
assign t_on_rough_minus10 = t_on_rough - 10;


reg t_off_end;
reg t_on_end;

always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) off_rough_amp <= 0;
    else begin
        if (refine_state_c == determine_t) begin
            if (cnt_check == t_off_rough) off_rough_amp <= amp_cur;
            else off_rough_amp <= off_rough_amp;
        end
        else if (refine_state_c == determine_p) begin
            if (cnt_check == p_off_rough) off_rough_amp <= amp_cur;
            else off_rough_amp <= off_rough_amp;            
        end
        else if (refine_state_c == refine_finish) begin
            off_rough_amp <= 0;
        end
    end
end
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) on_rough_amp <= 0;
    else begin
        if (refine_state_c == determine_t) begin
            if (cnt_check == t_on_rough) on_rough_amp <= amp_cur;
            else on_rough_amp <= on_rough_amp;
        end
        else if (refine_state_c == determine_p) begin
            if (cnt_check == p_on_rough) on_rough_amp <= amp_cur;
            else on_rough_amp <= on_rough_amp;            
        end
        else if (refine_state_c == refine_finish) begin
            on_rough_amp <= 0;
        end
    end
end
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        tp_max <= {1'B1,{(INPUT_DW-1){1'b0}}};
        tp_max_loc <= LENGTH_IN-1;
    end
    else begin
        if (refine_state_c == determine_t) begin
            if ((cnt_check > t_on_rough_minus1) & (cnt_check <  t_off_rough_plus1)) begin
                if (tp_max < amp_cur) begin
                    tp_max <= amp_cur;
                    tp_max_loc <= cnt_check;
                end
                else begin
                    tp_max <= tp_max;
                    tp_max_loc <= tp_max_loc;                    
                end
            end
            else begin
                tp_max <= tp_max;
                tp_max_loc <= tp_max_loc;                    
            end
        end
        else if (refine_state_c == select_rough_p) begin //reset
            tp_max <= {1'B1,{(INPUT_DW-1){1'b0}}};
            tp_max_loc <= LENGTH_IN-1;            
        end
        else if (refine_state_c == determine_p) begin
            if ((cnt_check > p_on_rough_minus1) & (cnt_check <  p_off_rough_plus1)) begin
                if (tp_max < amp_cur) begin
                    tp_max <= amp_cur;
                    tp_max_loc <= cnt_check;
                end
                else begin
                    tp_max <= tp_max;
                    tp_max_loc <= tp_max_loc;                    
                end
            end
            else begin
                tp_max <= tp_max;
                tp_max_loc <= tp_max_loc;                    
            end            
        end
        else if (refine_state_c == refine_finish) begin
            tp_max <= {1'B1,{(INPUT_DW-1){1'b0}}};
            tp_max_loc <= LENGTH_IN-1;
        end
    end
end
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        t_pos_temp <= {1'B1,{(INPUT_DW-1){1'b0}}};
        t_pos_loc_temp <= LENGTH_IN -1;
        t_neg_temp <= {1'B0,{(INPUT_DW-1){1'b1}}};
        t_neg_loc_temp <= LENGTH_IN -1;

    end
    else begin // lack reset
        if (refine_state_c == determine_t) begin
            if ((cnt_check > t_on_rough_minus1) & (cnt_check <  t_off_rough_plus1)) begin
                if ((trend == 5) |
                     ((trend == 2) & (trend_d == 3))|
                     ((trend == 2) & (trend_d == 0) & (trend_2d == 3))|
                     ((trend == 2) & (trend_d == 0) & (trend_2d == 0) & (trend_3d == 3))|
                     ((trend == 2) & (trend_d == 0) & (trend_2d == 0) & (trend_3d == 0) & (trend_4d == 3))|
                     ((trend == 2) & (trend_d == 0) & (trend_2d == 0) & (trend_3d == 0) & (trend_4d == 0) & (trend_5d == 3))) begin
                    if (amp_cur > t_pos_temp)  begin
                        t_pos_loc_temp <= cnt_check;
                        t_pos_temp <= amp_cur;
                    end
                    else begin
                        t_pos_loc_temp <= t_pos_loc_temp;
                        t_pos_temp <= t_pos_temp;                        
                    end
                end
            
                else if ((trend == 7) |
                        ((trend == 1) & (trend_d == 6))|
                        ((trend == 1) & (trend_d == 0) & (trend_2d == 6))|
                        ((trend == 1) & (trend_d == 0) & (trend_2d == 0) & (trend_3d == 6))|
                        ((trend == 1) & (trend_d == 0) & (trend_2d == 0) & (trend_3d == 0) & (trend_4d == 6))|
                        ((trend == 1) & (trend_d == 0) & (trend_2d == 0) & (trend_3d == 0) & (trend_4d == 0) & (trend_5d == 6))) begin
                    if (amp_cur < t_neg_temp)  begin
                        t_neg_loc_temp <= cnt_check;
                        t_neg_temp <= amp_cur;
                    end
                    else begin
                        t_neg_loc_temp <= t_neg_loc_temp;
                        t_neg_temp <= t_neg_temp;                        
                    end                    
                end
            end
            else begin
                t_pos_loc_temp <= t_pos_loc_temp;
                t_pos_temp <= t_pos_temp;   
                t_neg_loc_temp <= t_neg_loc_temp;
                t_neg_temp <= t_neg_temp;   
            end
        end
        
        else begin
            if (feature_done) begin
                t_pos_temp <= {1'B1,{(INPUT_DW-1){1'b0}}};
                t_pos_loc_temp <= LENGTH_IN -1;
                t_neg_temp <= {1'B0,{(INPUT_DW-1){1'b1}}};
                t_neg_loc_temp <= LENGTH_IN -1;              
            end
            else begin
                t_pos_loc_temp <= t_pos_loc_temp;
                t_pos_temp <= t_pos_temp;   
                t_neg_loc_temp <= t_neg_loc_temp;
                t_neg_temp <= t_neg_temp;     
            end 
        
        end
    end
end
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        t_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
        t_loc <= LENGTH_IN -1;
        t_dir <= 0;        
    end
    else begin
        if ((refine_state_c == determine_t) & (cnt_check == LENGTH_IN -1)) begin
            if ((t_pos_temp == {1'B1,{(INPUT_DW-1){1'b0}}}) & (t_neg_temp == {1'B0,{(INPUT_DW-1){1'b1}}})) begin
                t_amp <= tp_max;
                t_loc <= tp_max_loc;
                t_dir <= 1;
            end
            else if ((t_pos_temp != {1'B1,{(INPUT_DW-1){1'b0}}}) & (t_neg_temp == {1'B0,{(INPUT_DW-1){1'b1}}})) begin
                t_amp <= t_pos_temp;
                t_loc <= t_pos_loc_temp;
                t_dir  <= 1;      
            end
            else if ((t_pos_temp == {1'B1,{(INPUT_DW-1){1'b0}}}) & (t_neg_temp != {1'B0,{(INPUT_DW-1){1'b1}}})) begin
                t_amp <= t_neg_temp;
                t_loc <= t_neg_loc_temp;
                t_dir  <= -1;       
            end
            else  begin
                if ((t_pos_temp - off_rough_amp)   >= (off_rough_amp - t_neg_temp + 5))        begin
                    t_amp <= t_pos_temp;
                    t_loc <= t_pos_loc_temp;
                    t_dir  <= 1;               
                end
                else begin
                    t_amp <= t_neg_temp;
                    t_loc <= t_neg_loc_temp;
                    t_dir  <= -1;              
                end
            end
        end
        else begin
            if (mode == 0) begin
                if (ann_done) begin //ann need to use t_dir
                    t_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
                    t_loc <= LENGTH_IN -1;
                    t_dir <= 0;            
                end
                else begin
                    t_amp <= t_amp;
                    t_loc <= t_loc;
                    t_dir <= t_dir;      
                end
            end
            else begin
                if (feature_done) begin
                    t_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
                    t_loc <= LENGTH_IN -1;
                    t_dir <= 0;            
                end
                else begin
                    t_amp <= t_amp;
                    t_loc <= t_loc;
                    t_dir <= t_dir;      
                end
            end
        end
    end
end

always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        t_off_loc <= LENGTH_IN -1;
        t_off_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
        t_off_end <= 0;
    end
    else begin
        if (refine_state_c == determine_t_off)  begin // need reset
            if ((cnt_check == LENGTH_IN-1)&(!t_off_end)) begin
                t_off_loc <= t_off_rough;
                t_off_amp <= off_rough_amp;
                t_off_end <= 1; 
            end
            else begin
                if ((t_pos_temp != {1'B1,{(INPUT_DW-1){1'b0}}}) & (t_dir == 1)) begin
                    if ((cnt_check >= t_pos_loc_temp ) & (cnt_check <t_off_rough_plus1 ) & (!t_off_end) ) begin
                        if ((trend == 6)& (cnt_check - t_pos_loc_temp > t_pos_loc_temp -t_off_rough-5)  )    begin
                            t_off_loc <= cnt_check;
                            t_off_amp <= amp_cur;  
                            t_off_end <= 1;                      
                        end       
                        else begin
                            t_off_loc <= t_off_loc;
                            t_off_amp <= t_off_amp;
                            t_off_end <= t_off_end;                          
                        end       
                    end
                    else if ((cnt_check > t_off_rough) & (cnt_check < t_off_rough + 16) & (!t_off_end))begin
                        if (trend == 6) begin
                            t_off_loc <= cnt_check;
                            t_off_amp <= amp_cur;  
                            t_off_end <= 1;
                        end                 
                    
                        else begin
                            t_off_loc <= t_off_loc;
                            t_off_amp <= t_off_amp;
                            t_off_end <= t_off_end;                 
                        end
                    end
                    else;
                end
                else if  ((t_neg_temp != {1'B1,{(INPUT_DW-1){1'b0}}}) & (t_dir == -1)) begin
                    if ((cnt_check >= t_neg_loc_temp ) & (cnt_check <t_off_rough_plus1 ) & (!t_off_end) ) begin
                        if ((trend == 3)& (cnt_check - t_neg_loc_temp > t_neg_loc_temp -t_off_rough-3)  )    begin
                            t_off_loc <= cnt_check;
                            t_off_amp <= amp_cur;  
                            t_off_end <= 1;                      
                        end       
                        else begin
                            t_off_loc <= t_off_loc;
                            t_off_amp <= t_off_amp;
                            t_off_end <= t_off_end;                          
                        end       
                    end
                    else if ((cnt_check > t_off_rough) & (cnt_check < t_off_rough + 16) & (!t_off_end))begin
                        if (trend == 3) begin
                            t_off_loc <= cnt_check;
                            t_off_amp <= amp_cur;  
                            t_off_end <= 1;                 
                        end
                        else begin
                            t_off_loc <= t_off_loc;
                            t_off_amp <= t_off_amp;
                            t_off_end <= t_off_end;                 
                        end                
                    end
                    else;
                end

                else begin
                    t_off_loc <= t_off_loc;
                    t_off_amp <= t_off_amp;
                    t_off_end <= t_off_end;                   
                end
            end
        end
        else begin
            if (feature_done) begin
                t_off_loc <= LENGTH_IN -1;
                t_off_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
                t_off_end <= 0;           
            end
            else begin
                t_off_loc <= t_off_loc;
                t_off_amp <= t_off_amp;
                t_off_end <= t_off_end;    
            end       
        end
    end
end

always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        t_on_loc <= LENGTH_IN -1;
        t_on_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
        t_on_end <= 0;
    end
    else begin
        if (refine_state_c == determine_t_on)  begin // need reset
            if ((cnt_check == LENGTH_IN-1)&(!t_on_end)) begin
                    t_on_loc <= t_on_rough;
                    t_on_amp <= on_rough_amp;
                    t_on_end <= 1;                      
            end
            else begin
                if ((t_pos_temp != {1'B1,{(INPUT_DW-1){1'b0}}}) & (t_dir == 1)) begin
                    if ((cnt_check_com > t_on_rough_minus1 ) & (cnt_check_com <t_pos_loc_temp )  & (!t_on_end)) begin
                        if (((trend == 1)|(trend == 7)) & ( t_pos_loc_temp-cnt_check  > t_pos_loc_temp -t_on_rough-3)  )    begin
                            t_on_loc <= cnt_check_com;
                            t_on_amp <= amp_cur;  
                            t_on_end <= 1;                      
                        end       
                        else begin
                            t_on_loc <= t_on_loc;
                            t_on_amp <= t_on_amp;
                            t_on_end <= t_on_end;                          
                        end       
                    end
                    else if ((cnt_check_com > t_on_rough_minus10) & (cnt_check_com < t_on_rough_minus1) & (!t_on_end))begin
                        if ((trend == 1)|(trend == 7)) begin
                            t_on_loc <= cnt_check_com;
                            t_on_amp <= amp_cur;  
                            t_on_end <= 1;
                        end                 
                    
                        else begin
                            t_on_loc <= t_on_loc;
                            t_on_amp <= t_on_amp;
                            t_on_end <= t_on_end;                 
                        end
                    end
                    else;
                end
                else if  ((t_neg_temp != {1'B1,{(INPUT_DW-1){1'b0}}}) & (t_dir == -1)) begin
                    if ((cnt_check_com > t_on_rough_minus1 ) & (cnt_check_com <t_neg_loc_temp ) & (!t_on_end)) begin
                        if (((trend == 2)|(trend == 5) )&( t_neg_loc_temp-cnt_check  > t_neg_loc_temp -t_on_rough-3) )    begin
                            t_on_loc <= cnt_check_com;
                            t_on_amp <= amp_cur;  
                            t_on_end <= 1;                      
                        end       
                        else begin
                            t_on_loc <= t_on_loc;
                            t_on_amp <= t_on_amp;
                            t_on_end <= t_on_end;                          
                        end       
                    end
                    else if ((cnt_check_com > t_on_rough_minus10) & (cnt_check_com < t_on_rough_minus1) & (!t_on_end))begin
                        if ((trend == 2)|(trend == 5)) begin
                            t_on_loc <= cnt_check_com;
                            t_on_amp <= amp_cur;  
                            t_on_end <= 1;
                        end                 
                    
                        else begin
                            t_on_loc <= t_on_loc;
                            t_on_amp <= t_on_amp;
                            t_on_end <= t_on_end;                 
                        end
                    end
                    else;
                end

                else begin
                    t_on_loc <= t_on_loc;
                    t_on_amp <= t_on_amp;
                    t_on_end <= t_on_end;                 
                end
            end
        end
        else begin
            if (feature_done) begin
                t_on_loc <= LENGTH_IN -1;
                t_on_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
                t_on_end <= 0;
            end
            else begin
                t_on_loc <= t_on_loc;
                t_on_amp <= t_on_amp;
                t_on_end <= t_on_end;                  
            end          
        end
    end
end


// determine p
reg signed [INPUT_DW-1:0] p_pos_temp;
reg [$clog2(LENGTH_IN+1)-1:0] p_pos_loc_temp;
reg signed [INPUT_DW-1:0] p_neg_temp;
reg [$clog2(LENGTH_IN+1)-1:0] p_neg_loc_temp;

reg p_off_end;
reg p_on_end;
assign p_on_rough_minus1 = p_on_rough - 1;
assign p_off_rough_plus1 = p_off_rough + 1;
assign p_off_rough_16 = p_off_rough + 16;
assign p_on_rough_minus10 = p_on_rough - 10;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        p_pos_temp <= {1'B1,{(INPUT_DW-1){1'b0}}};
        p_pos_loc_temp <= LENGTH_IN -1;
        p_neg_temp <= {1'B0,{(INPUT_DW-1){1'b1}}};
        p_neg_loc_temp <= LENGTH_IN -1;

    end
    else begin // lack reset
        if (refine_state_c == determine_p) begin
            if ((cnt_check > p_on_rough_minus1) & (cnt_check <  p_off_rough_plus1)) begin
                if ((trend == 5) |
                     ((trend == 2) & (trend_d == 3))|
                     ((trend == 2) & (trend_d == 0) & (trend_2d == 3))|
                     ((trend == 2) & (trend_d == 0) & (trend_2d == 0) & (trend_3d == 3))|
                     ((trend == 2) & (trend_d == 0) & (trend_2d == 0) & (trend_3d == 0) & (trend_4d == 3))|
                     ((trend == 2) & (trend_d == 0) & (trend_2d == 0) & (trend_3d == 0) & (trend_4d == 0) & (trend_5d == 3))) begin
                    if (amp_cur > p_pos_temp)  begin
                        p_pos_loc_temp <= cnt_check;
                        p_pos_temp <= amp_cur;
                    end
                    else begin
                        p_pos_loc_temp <= p_pos_loc_temp;
                        p_pos_temp <= p_pos_temp;                        
                    end
                end
            
                else if ((trend == 7) |
                        ((trend == 1) & (trend_d == 6))|
                        ((trend == 1) & (trend_d == 0) & (trend_2d == 6))|
                        ((trend == 1) & (trend_d == 0) & (trend_2d == 0) & (trend_3d == 6))|
                        ((trend == 1) & (trend_d == 0) & (trend_2d == 0) & (trend_3d == 0) & (trend_4d == 6))|
                        ((trend == 1) & (trend_d == 0) & (trend_2d == 0) & (trend_3d == 0) & (trend_4d == 0) & (trend_5d == 6))) begin
                    if (amp_cur < p_neg_temp)  begin
                        p_neg_loc_temp <= cnt_check;
                        p_neg_temp <= amp_cur;
                    end
                    else begin
                        p_neg_loc_temp <= p_neg_loc_temp;
                        p_neg_temp <= p_neg_temp;                        
                    end                    
                end
            end
            else begin
                p_pos_loc_temp <= p_pos_loc_temp;
                p_pos_temp <= p_pos_temp;   
                p_neg_loc_temp <= p_neg_loc_temp;
                p_neg_temp <= p_neg_temp;   
            end
        end
        
        else begin
            if (feature_done ) begin
                p_pos_temp <= {1'B1,{(INPUT_DW-1){1'b0}}};
                p_pos_loc_temp <= LENGTH_IN -1;
                p_neg_temp <= {1'B0,{(INPUT_DW-1){1'b1}}};
                p_neg_loc_temp <= LENGTH_IN -1;
            end
            else begin
                p_pos_loc_temp <= p_pos_loc_temp;
                p_pos_temp <= p_pos_temp;   
                p_neg_loc_temp <= p_neg_loc_temp;
                p_neg_temp <= p_neg_temp;                 
            end           
        end
    end
end
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        p_amp <=  {1'B1,{(INPUT_DW-1){1'b0}}};
        p_loc <= LENGTH_IN -1;
        p_dir <= 0;        
    end
    else begin
        if ((refine_state_c == determine_p) & (cnt_check == LENGTH_IN -1)) begin
            if ((p_pos_temp == {1'B1,{(INPUT_DW-1){1'b0}}}) & (p_neg_temp == {1'B0,{(INPUT_DW-1){1'b1}}})) begin
                p_amp <= tp_max;
                p_loc <= tp_max_loc;
                p_dir <= 1;
            end
            else if ((p_pos_temp != {1'B1,{(INPUT_DW-1){1'b0}}}) & (p_neg_temp == {1'B0,{(INPUT_DW-1){1'b1}}})) begin
                p_amp <= p_pos_temp;
                p_loc <= p_pos_loc_temp;
                p_dir  <= 1;      
            end
            else if ((p_pos_temp == {1'B1,{(INPUT_DW-1){1'b0}}}) & (p_neg_temp != {1'B0,{(INPUT_DW-1){1'b1}}})) begin
                p_amp <= p_neg_temp;
                p_loc <= p_neg_loc_temp;
                p_dir  <= -1;       
            end
            else  begin
                if ((p_pos_temp - off_rough_amp)   >= (off_rough_amp - p_neg_temp +5))        begin
                    p_amp <= p_pos_temp;
                    p_loc <= p_pos_loc_temp;
                    p_dir  <= 1;               
                end
                else begin
                    p_amp <= p_neg_temp;
                    p_loc <= p_neg_loc_temp;
                    p_dir  <= -1;              
                end
            end
        end
        else begin
            if (feature_done ) begin
                p_amp <=  {1'B1,{(INPUT_DW-1){1'b0}}};
                p_loc <= LENGTH_IN -1;
                p_dir <= 0;   
            end
            else begin
                p_amp <= p_amp;
                p_loc <= p_loc;
                p_dir <= p_dir;                  
            end               
                
        end
    end

end

always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        p_off_loc <= LENGTH_IN -1;
        p_off_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
        p_off_end <= 0;
    end
    else begin
        if (refine_state_c == determine_p_off)  begin // need reset
            if ((cnt_check == LENGTH_IN-1)&(!p_off_end)) begin
                p_off_loc <= p_off_rough;
                p_off_amp <= off_rough_amp;
                p_off_end <= 1;                    
            end
            else begin
                if ((p_pos_temp != {1'B1,{(INPUT_DW-1){1'b0}}}) & (p_dir == 1)) begin
                    if ((cnt_check >= p_pos_loc_temp ) & (cnt_check <p_off_rough_plus1 )& (!p_off_end) ) begin
                        if ((trend == 6)& (cnt_check - p_pos_loc_temp > p_pos_loc_temp -p_off_rough -5)  )    begin
                            p_off_loc <= cnt_check;
                            p_off_amp <= amp_cur;  
                            p_off_end <= 1;                      
                        end       
                        else begin
                            p_off_loc <= p_off_loc;
                            p_off_amp <= p_off_amp;
                            p_off_end <= p_off_end;                          
                        end       
                    end
                    else if ((cnt_check > p_off_rough) & (cnt_check < p_off_rough + 16) & (!p_off_end))begin
                        if (trend == 6) begin
                            p_off_loc <= cnt_check;
                            p_off_amp <= amp_cur;  
                            p_off_end <= 1;
                        end                 
                    
                        else begin
                            p_off_loc <= p_off_loc;
                            p_off_amp <= p_off_amp;
                            p_off_end <= p_off_end;                 
                        end
                    end
                    else;
                end
                else if  ((p_neg_temp != {1'B1,{(INPUT_DW-1){1'b0}}}) & (p_dir == -1)) begin
                    if ((cnt_check >= p_neg_loc_temp ) & (cnt_check <p_off_rough_plus1 ) & (!p_off_end)) begin
                        if ((trend == 3)& (cnt_check - p_neg_loc_temp > p_neg_loc_temp -p_off_rough -3)  )    begin
                            p_off_loc <= cnt_check;
                            p_off_amp <= amp_cur;  
                            p_off_end <= 1;                      
                        end       
                        else begin
                            p_off_loc <= p_off_loc;
                            p_off_amp <= p_off_amp;
                            p_off_end <= p_off_end;                          
                        end       
                    end
                    else if ((cnt_check > p_off_rough) & (cnt_check < p_off_rough + 16) & (!p_off_end))begin
                        if (trend == 3) begin
                            p_off_loc <= cnt_check;
                            p_off_amp <= amp_cur;  
                            p_off_end <= 1;                 
                        end
                        else begin
                            p_off_loc <= p_off_loc;
                            p_off_amp <= p_off_amp;
                            p_off_end <= p_off_end;                 
                        end                
                    end

                    else;
                end

                else begin
                    p_off_loc <= p_off_loc;
                    p_off_amp <= p_off_amp;
                    p_off_end <= p_off_end;                 
                end 
            end

        end
        else begin
            if (feature_done) begin
                p_off_loc <= LENGTH_IN -1;
                p_off_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
                p_off_end <= 0;                
            end
            else begin
                p_off_loc <= p_off_loc;
                p_off_amp <= p_off_amp;
                p_off_end <= p_off_end;                         
            end
        end
    end
end

always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        p_on_loc <= LENGTH_IN -1;
        p_on_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
        p_on_end <= 0;
    end
    else begin
        if (refine_state_c == determine_p_on)  begin // need reset
            if ((cnt_check == LENGTH_IN-1)&(!p_on_end)) begin
                p_on_loc <= p_on_rough;
                p_on_amp <= on_rough_amp;
                p_on_end <= 1;   
            end
            else begin
                if ((p_pos_temp != {1'B1,{(INPUT_DW-1){1'b0}}}) & (p_dir == 1)) begin
                    if ((cnt_check_com > p_on_rough_minus1 ) & (cnt_check_com <p_pos_loc_temp )& (!p_on_end) ) begin
                        if (((trend == 1)|(trend == 7) )&( p_pos_loc_temp-cnt_check  > p_pos_loc_temp -p_on_rough-3)  )    begin
                            p_on_loc <= cnt_check_com;
                            p_on_amp <= amp_cur;  
                            p_on_end <= 1;                      
                        end       
                        else begin
                            p_on_loc <= p_on_loc;
                            p_on_amp <= p_on_amp;
                            p_on_end <= p_on_end;                          
                        end       
                    end
                    else if ((cnt_check_com > p_on_rough_minus10) & (cnt_check_com < p_on_rough_minus1) & (!p_on_end))begin
                        if ((trend == 1)|(trend == 7)) begin
                            p_on_loc <= cnt_check_com;
                            p_on_amp <= amp_cur;  
                            p_on_end <= 1;
                        end                 
                    
                        else begin
                            p_on_loc <= p_on_loc;
                            p_on_amp <= p_on_amp;
                            p_on_end <= p_on_end;                 
                        end
                    end
                    else; 
                end
                else if  ((p_neg_temp != {1'B1,{(INPUT_DW-1){1'b0}}}) & (p_dir == -1)) begin
                    if ((cnt_check_com > p_on_rough_minus1 ) & (cnt_check_com <p_neg_loc_temp )& (!p_on_end) ) begin
                        if (((trend == 2)|(trend == 5) )&( p_neg_loc_temp-cnt_check  > p_neg_loc_temp -p_on_rough-3)  )    begin
                            p_on_loc <= cnt_check_com;
                            p_on_amp <= amp_cur;  
                            p_on_end <= 1;                      
                        end       
                        else begin
                            p_on_loc <= p_on_loc;
                            p_on_amp <= p_on_amp;
                            p_on_end <= p_on_end;                          
                        end       
                    end
                    else if ((cnt_check_com > p_on_rough_minus10) & (cnt_check_com < p_on_rough_minus1) & (!p_on_end))begin
                        if ((trend == 2)|(trend == 5)) begin
                            p_on_loc <= cnt_check_com;
                            p_on_amp <= amp_cur;  
                            p_on_end <= 1;
                        end                 
                    
                        else begin
                            p_on_loc <= p_on_loc;
                            p_on_amp <= p_on_amp;
                            p_on_end <= p_on_end;                 
                        end
                    end
                    else; 
                end

                else begin
                    p_on_loc <= p_on_loc;
                    p_on_amp <= p_on_amp;
                    p_on_end <= p_on_end;                 
                end
            end
        end
        else begin
            if (feature_done) begin
                p_on_loc <= LENGTH_IN -1;
                p_on_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
                p_on_end <= 0;                
            end
            else begin
                p_on_loc <= p_on_loc;
                p_on_amp <= p_on_amp;
                p_on_end <= p_on_end;                  
            end
          
        end
    end
end
// pq loc
reg pq_end;
reg signed [INPUT_DW-1:0] q_on_rough_amp;
wire [$clog2(LENGTH_IN+1)-1:0] r_loc_minus15;
assign r_loc_minus15 = r_loc - 15;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) q_on_rough_amp <= 0;
    else begin
        if (refine_state_c == determine_r) begin
            if (cnt_check == q_on_rough) q_on_rough_amp <= amp_cur;
            else q_on_rough_amp <= q_on_rough_amp;
        end
        else if (refine_state_c == refine_finish) begin
            q_on_rough_amp <= 0;
        end
    end
end
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        pq_loc <= LENGTH_IN -1;
        pq_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
        pq_end <= 0;
    end
    else begin
        if (refine_state_c == determine_q) begin
            if ((cnt_check >= p_off_loc) & (cnt_check < r_loc ) & (!pq_end)) begin
                if ((trend == 2) | (trend == 5))  begin
                    pq_loc <= cnt_check;
                    pq_amp <= amp_cur;
                    pq_end <= 1;
                end
                else begin
                    pq_loc <= pq_loc;
                    pq_amp <= pq_amp;
                    pq_end <= pq_end;                    
                end
            end
            else begin
                pq_loc <= pq_loc;
                pq_amp <= pq_amp;
                pq_end <= pq_end;                  
            end
        end
        else if (refine_state_c == determine_q_re) begin
            if ((cnt_check == LENGTH_IN-1)& (!pq_end)) begin
                pq_loc <= q_on_rough;
                pq_amp <= q_on_rough_amp;
                pq_end <= 1;                
            end
            else begin
                

                if ((cnt_check_com >= r_loc_minus15) & (cnt_check_com < r_loc ) & (!pq_end)) begin
                    if ((trend == 7) | (trend == 6))  begin
                        pq_loc <= cnt_check_com;
                        pq_amp <= amp_cur;
                        pq_end <= 1;
                    end
                    else begin
                        pq_loc <= pq_loc;
                        pq_amp <= pq_amp;
                        pq_end <= pq_end;                    
                    end
                end
                else begin
                    pq_loc <= pq_loc;
                    pq_amp <= pq_amp;
                    pq_end <= pq_end;                  
                end        
            end    
        end
        else begin
            if (feature_done) begin
                pq_loc <= LENGTH_IN -1;
                pq_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
                pq_end <= 0;                
            end
            else begin
                pq_loc <= pq_loc;
                pq_amp <= pq_amp;
                pq_end <= pq_end;        
            end          
        end
    end
end
// q loc
reg q_end;

always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        q_loc <= LENGTH_IN -1;
        q_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
        q_end <= 0;
    end
    else begin
        if (refine_state_c == determine_q_re) begin
            if ((cnt_check == LENGTH_IN - 1) & (!q_end)) begin
                q_loc <= q_on_rough;
                q_amp <= q_on_rough_amp;
                q_end <= 1;                
            end
            else begin
                if (p_off_loc != LENGTH_IN - 1) begin // p exist
                    if ((cnt_check_com >= p_off_loc) & (cnt_check_com < r_loc ) & (!q_end)) begin
                        if ((trend == 7) | (trend == 6))  begin
                            q_loc <= cnt_check_com;
                            q_amp <= amp_cur;
                            q_end <= 1;
                        end
                        else begin
                            q_loc <= q_loc;
                            q_amp <= q_amp;
                            q_end <= q_end;                    
                        end
                    end
                    else begin
                        q_loc <= q_loc;
                        q_amp <= q_amp;
                        q_end <= q_end;                  
                    end    
                end
                else begin
                    if ((cnt_check_com >= r_loc_minus15) & (cnt_check_com < r_loc ) & (!q_end)) begin
                        if ((trend == 7) | (trend == 6))  begin
                            q_loc <= cnt_check_com;
                            q_amp <= amp_cur;
                            q_end <= 1;
                        end
                        else begin
                            q_loc <= q_loc;
                            q_amp <= q_amp;
                            q_end <= q_end;                    
                        end
                    end
                    else begin
                        q_loc <= q_loc;
                        q_amp <= q_amp;
                        q_end <= q_end;                  
                    end                 
                end 
            end       
        end
        else begin
            if (feature_done) begin
                q_loc <= LENGTH_IN -1;
                q_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
                q_end <= 0;              
            end
            else begin
                q_loc <= q_loc;
                q_amp <= q_amp;
                q_end <= q_end;                  
            end
                  
        end
    end
end
// s_loc
reg s_end;
reg signed [INPUT_DW-1:0] s_off_rough_amp;
wire [$clog2(LENGTH_IN+1)-1:0] r_loc_plus30;
assign r_loc_plus30 = r_loc + 30;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) s_off_rough_amp <= 0;
    else begin
        if (refine_state_c == determine_r) begin
            if (cnt_check == s_off_rough) s_off_rough_amp <= amp_cur;
            else s_off_rough_amp <= s_off_rough_amp;
        end
        else if (refine_state_c == refine_finish) begin
            s_off_rough_amp <= 0;
        end
    end
end
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        s_loc <= LENGTH_IN -1;
        s_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
        s_end <= 0;
    end
    else begin
        if (refine_state_c == determine_s) begin
            if (t_on_loc != LENGTH_IN - 1) begin
                if ((cnt_check > r_loc) & (cnt_check <= t_on_loc ) & (!s_end)) begin
                    if ((trend == 7) | (trend == 6))  begin
                        s_loc <= cnt_check;
                        s_amp <= amp_cur;
                        s_end <= 1;
                    end
                    else begin
                        s_loc <= s_loc;
                        s_amp <= s_amp;
                        s_end <= s_end;                    
                    end
                end
                else begin
                    s_loc <= s_loc;
                    s_amp <= s_amp;
                    s_end <= s_end;                  
                end                
            end
            else begin
                if ((cnt_check > r_loc) & (cnt_check <= r_loc_plus30 ) & (!s_end)) begin
                    if ((trend == 7) | (trend == 6))  begin
                        s_loc <= cnt_check;
                        s_amp <= amp_cur;
                        s_end <= 1;
                    end
                    else begin
                        s_loc <= s_loc;
                        s_amp <= s_amp;
                        s_end <= s_end;                    
                    end
                end
                else begin
                    s_loc <= s_loc;
                    s_amp <= s_amp;
                    s_end <= s_end;                  
                end                   
            end

        end
        else if (refine_state_c == s_st_modify) begin
            if (s_loc == LENGTH_IN -1) begin
                s_loc <= s_off_rough;
                s_amp <= s_off_rough_amp;
                s_end <= 1;
            end    
            else begin
                s_loc <= s_loc;
            end        
        end
        else begin
            if (feature_done) begin
                s_loc <= LENGTH_IN -1;
                s_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
                s_end <= 0;                
            end
            else begin
                s_loc <= s_loc;
                s_amp <= s_amp;
                s_end <= s_end;                   
            end
             
        end
    end
end
//st_loc
reg st_end;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        st_loc <= LENGTH_IN -1;
        st_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
        st_end <= 0;
    end
    else begin
        if (refine_state_c == determine_s) begin
            if (t_on_loc == LENGTH_IN - 1) begin
                if ((cnt_check > r_loc) & (cnt_check <= r_loc_plus30 ) & (!st_end)) begin
                    if ((trend == 3) | (trend == 5))  begin
                        st_loc <= cnt_check;
                        st_amp <= amp_cur;
                        st_end <= 1;
                    end
                    else begin
                        st_loc <= st_loc;
                        st_amp <= st_amp;
                        st_end <= st_end;                    
                    end
                end
                else begin
                    st_loc <= st_loc;
                    st_amp <= st_amp;
                    st_end <= st_end;                  
                end                
            end
            else begin
                st_loc <= st_loc;
                st_amp <= st_amp;
                st_end <= st_end;                  
            end
        end
        else if (refine_state_c == determine_s_re) begin
            if ((cnt_check_com > r_loc) & (cnt_check_com <= t_on_loc ) & (!st_end)) begin
                if ((trend == 3) | (trend == 5))  begin
                    st_loc <= cnt_check_com;
                    st_amp <= amp_cur;
                    st_end <= 1;
                end
                else begin
                    st_loc <= st_loc;
                    st_amp <= st_amp;
                    st_end <= st_end;                    
                end
            end
            else begin
                st_loc <= st_loc;
                st_amp <= st_amp;
                st_end <= st_end;                  
            end              
        end
        else if (refine_state_c == s_st_modify) begin
            if ((st_loc == LENGTH_IN -1)) begin
                if (s_loc != LENGTH_IN -1) begin
                    st_loc <= s_loc;
                    st_amp <= s_amp;
                    st_end <= 1;
                end
                else begin
                    st_loc <= s_off_rough;
                    st_amp <= s_off_rough_amp;
                    st_end <= 1;                    
                end
            end    
            else begin
                st_loc <= st_loc;
                st_amp <= st_amp;
                st_end <= st_end;
            end        
        end
        else begin
            if (feature_done) begin
                st_loc <= LENGTH_IN -1;
                st_amp <= {1'B1,{(INPUT_DW-1){1'b0}}};
                st_end <= 0;                
            end
            else begin
                st_loc <= st_loc;
                st_amp <= st_amp;
                st_end <= st_end;                  
            end
            
        end
    end
end
// mi_points
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        iso_line <= 0;
    end
    else begin
        if (refine_state_c == mi_points) begin
            if (p_loc != LENGTH_IN -1) begin // p_exsit
                if (cnt_check == (p_on_loc - 10) ) iso_line <= amp_cur;
                else iso_line <= iso_line;
            end
            else iso_line <= 0;
        end
    end
end
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        st_amp_1 <= 0;
        st_amp_2 <= 0;
        st_amp_4 <= 0;
        st_amp_6 <= 0;
    end
    else begin
        if (refine_state_c == mi_points) begin
            if ((t_loc != LENGTH_IN -1) & (st_loc < LENGTH_IN -1 -6) & (r_loc != LENGTH_IN-1)) begin // t_exsit
                if (cnt_check == st_loc  + 1) st_amp_1 <= amp_cur;
                else if (cnt_check == st_loc  + 2) st_amp_2 <= amp_cur;
                else if (cnt_check == st_loc  + 4) st_amp_4 <= amp_cur;
                else if (cnt_check == st_loc + 6)  st_amp_6 <= amp_cur;
                else begin
                    st_amp_1 <= st_amp_1;
                    st_amp_2 <= st_amp_2;
                    st_amp_4 <= st_amp_4;
                    st_amp_6 <= st_amp_6;                    
                end
            end
            else begin
                st_amp_1 <= 0;
                st_amp_2 <= 0;
                st_amp_4 <= 0;
                st_amp_6 <= 0;                
            end
        end
    end
end
// embedding
localparam  EMB_QRS_THRES = 8;
localparam  EMB_T_THRES = 4;
reg  [$clog2(QRS_EMB_LEN+1)-1: 0] cnt_valid_emb_qrs;
reg  [$clog2(T_EMB_LEN+1)-1: 0] cnt_valid_emb_t;
// wire [INTEVAL_DW-1: 0] qrs_mi;
// wire [INTEVAL_DW-1: 0] t_dur_mi;
wire signed [EMB_DW -1 : 0] emb_val_qrs;
wire signed [EMB_DW -1 : 0] emb_val_t;
// assign qrs_mi = (post_state_c == embedding)?(s_loc - q_loc):0;
// assign  t_dur_mi = (post_state_c == embedding)?(t_off_loc - t_on_loc):0;
assign emb_shift = (post_state_c == embedding);
assign emb_val_qrs = (post_state_c == embedding)? ((amp_pre_minus_cur > EMB_QRS_THRES) ? 2'B11:((amp_pre_minus_cur  < -EMB_QRS_THRES)? 2'B01: 2'B00)):2'B00; // need enable
assign emb_val_t = (post_state_c == embedding)? ((amp_pre_minus_cur > EMB_T_THRES) ? 2'B11:((amp_pre_minus_cur  < -EMB_T_THRES)? 2'B01: 2'B00)):2'B00; // need enable


always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        qrs_emb_buffer <= 0;
        cnt_valid_emb_qrs <= 0;
    end
    else begin
        if (post_state_c == embedding) begin
            if ((cnt_check >  q_loc) & (cnt_check <  s_loc)) begin
                if (cnt_valid_emb_qrs < QRS_EMB_LEN) begin
                    if (emb_val_qrs != 2'B00) begin
                        qrs_emb_buffer <= {emb_val_qrs, qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1:EMB_DW]};
                        cnt_valid_emb_qrs <= cnt_valid_emb_qrs + 1;
                    end
                    else begin
                        qrs_emb_buffer <= qrs_emb_buffer;
                        cnt_valid_emb_qrs <= cnt_valid_emb_qrs;                           
                    end
                end
                else begin
                    qrs_emb_buffer <= qrs_emb_buffer;
                    cnt_valid_emb_qrs <= cnt_valid_emb_qrs;                    
                end
            end
            else if (cnt_check >= s_loc) begin
                if (cnt_valid_emb_qrs < QRS_EMB_LEN) begin
                    case(cnt_valid_emb_qrs)
                    0: qrs_emb_buffer <= {{(EMB_DW*QRS_EMB_LEN){1'B0}}};
                    1: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-1)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:1* EMB_DW]};
                    2: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-2)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:2* EMB_DW]};
                    3: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-3)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:3* EMB_DW]};
                    4: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-4)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:4* EMB_DW]};
                    5: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-5)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:5* EMB_DW]};
                    6: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-6)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:6* EMB_DW]};
                    7: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-7)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:7* EMB_DW]};
                    8: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-8)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:8* EMB_DW]};
                    9: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-9)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:9* EMB_DW]};
                    10: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-10)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:10* EMB_DW]};
                    11: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-11)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:11* EMB_DW]};
                    12: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-12)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:12* EMB_DW]};
                    13: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-13)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:13* EMB_DW]};
                    14: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-14)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:14* EMB_DW]};
                    15: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-15)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:15* EMB_DW]};
                    16: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-16)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:16* EMB_DW]};
                    17: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-17)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:17* EMB_DW]};
                    18: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-18)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:18* EMB_DW]};
                    19: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-19)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:19* EMB_DW]};
                    20: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-20)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:20* EMB_DW]};
                    21: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-21)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:21* EMB_DW]};
                    22: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-22)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:22* EMB_DW]};
                    23: qrs_emb_buffer <= {{(EMB_DW*(QRS_EMB_LEN-23)){1'B0}},qrs_emb_buffer[ EMB_DW*QRS_EMB_LEN-1-:23* EMB_DW]};
                    endcase
                    cnt_valid_emb_qrs <= QRS_EMB_LEN;
                end
                else begin
                    qrs_emb_buffer <= qrs_emb_buffer;
                    cnt_valid_emb_qrs <= cnt_valid_emb_qrs;                     
                end
                
            end
        end
        else if (post_state_c == done) begin //RST
            qrs_emb_buffer <= qrs_emb_buffer;
            cnt_valid_emb_qrs <= 0;                
        end
    end
end
                
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        t_emb_buffer <= 0;
        cnt_valid_emb_t <= 0;
    end
    else begin
        if (post_state_c == embedding) begin
            if ((cnt_check >  t_on_loc) & (cnt_check <  t_off_loc)) begin
                if (cnt_valid_emb_t < T_EMB_LEN) begin
                    if (emb_val_t != 2'b00) begin
                        t_emb_buffer <= {emb_val_t, t_emb_buffer[ EMB_DW*T_EMB_LEN-1:EMB_DW]};
                        cnt_valid_emb_t <= cnt_valid_emb_t + 1;
                    end
                    else begin
                        t_emb_buffer <= t_emb_buffer;
                        cnt_valid_emb_t <= cnt_valid_emb_t;                              
                    end
                end
                else begin
                    t_emb_buffer <= t_emb_buffer;
                    cnt_valid_emb_t <= cnt_valid_emb_t;                    
                end
            end
            else if (cnt_check >= t_off_loc) begin
                if (cnt_valid_emb_t < T_EMB_LEN) begin
                    case(cnt_valid_emb_t)
                    0: t_emb_buffer <= {{(EMB_DW*T_EMB_LEN){1'B0}}};
                    1: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-1)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:1* EMB_DW]};
                    2: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-2)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:2* EMB_DW]};
                    3: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-3)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:3* EMB_DW]};
                    4: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-4)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:4* EMB_DW]};
                    5: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-5)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:5* EMB_DW]};
                    6: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-6)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:6* EMB_DW]};
                    7: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-7)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:7* EMB_DW]};
                    8: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-8)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:8* EMB_DW]};
                    9: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-9)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:9* EMB_DW]};
                    10: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-10)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:10* EMB_DW]};
                    11: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-11)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:11* EMB_DW]};
                    12: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-12)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:12* EMB_DW]};
                    13: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-13)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:13* EMB_DW]};
                    14: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-14)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:14* EMB_DW]};
                    15: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-15)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:15* EMB_DW]};
                    16: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-16)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:16* EMB_DW]};
                    17: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-17)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:17* EMB_DW]};
                    18: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-18)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:18* EMB_DW]};
                    19: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-19)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:19* EMB_DW]};
                    20: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-20)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:20* EMB_DW]};
                    21: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-21)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:21* EMB_DW]};
                    22: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-22)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:22* EMB_DW]};
                    23: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-23)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:23* EMB_DW]};
                    24: t_emb_buffer <= {{(EMB_DW*(T_EMB_LEN-24)){1'B0}},t_emb_buffer[ EMB_DW*T_EMB_LEN-1-:24* EMB_DW]};
                    endcase
                    cnt_valid_emb_t <= T_EMB_LEN;
                end
                else begin
                    t_emb_buffer <= t_emb_buffer;
                    cnt_valid_emb_t <= cnt_valid_emb_t;                     
                end
                
            end
        end
        else if (post_state_c == done) begin //RST
            t_emb_buffer <= t_emb_buffer;
            cnt_valid_emb_t <= 0;                
        end
    end
end
           

// always @(posedge wclk or negedge rst_n) begin
//     if (!rst_n) begin
//         t_emb_buffer <= 0;
//     end
//     else begin
//         if (post_state_c == embedding) begin
            
//             if (t_dur_mi == 0) t_emb_buffer <= 0;
//             else if (t_dur_mi <= T_EMB_LEN) begin
//                 if ((cnt_check >  t_on_loc) & (cnt_check <  t_off_loc)) begin
//                     t_emb_buffer <= {emb_val_t, t_emb_buffer[ EMB_DW*T_EMB_LEN-1:EMB_DW]};
//                 end
//                 else t_emb_buffer <= t_emb_buffer;
//             end
//             else if (t_dur_mi > T_EMB_LEN)  begin
//                 if ((cnt_check >  t_on_loc) & (cnt_check < t_on_loc + T_EMB_LEN)) begin
//                     t_emb_buffer <= {emb_val_t, t_emb_buffer[EMB_DW*T_EMB_LEN-1:EMB_DW]};
//                 end
//                 else t_emb_buffer <= t_emb_buffer;
//             end
//         end
//         else t_emb_buffer <= t_emb_buffer;
//     end
// end

endmodule