`timescale  1ns/100ps
// 22*32+ 32*5+ 32*1+5*1 = 901, （65*32+32*2+32+2）*12 = 26136 
module ANN #(parameter INPUT_DW = 12,
    DATA_DW = 8,
    FEATURE_SUM_DW = INPUT_DW + 4,
    LENGTH_IN = 256,
    SRAM1024_AW = 10,
    SRAM8192_AW = 13,
    SRAM512_AW = 9,
    SRAM16_DW = 16,
    ARR_LABEL_DW = 3,
    DIR_DW = 2,
    SPAD_DEPTH = 8,
    INTEVAL_DW = $clog2(LENGTH_IN+1),
    NUM_FEAS_MI = 7,
    FEATURE_DIM = 22,
    FEATURE_DIM_MI = 64,
    ANN_HIDDEN_DIM = 32,
    ANN_OUT_DIM = 5,
    ANN_OUT_DIM_MI = 2,
    NUM_LEADS = 12,
    EMB_DW = 2,
    QRS_EMB_LEN = 24,
    T_EMB_LEN = 25,
    PARAM_DW = 13,
    ACTIVATION_BUF_LEN1 = 32*64,
    ACTIVATION_BUF_LEN3 = 32*64,
    ACTIVATION_BUF_LEN4 = 64*64
)(
    input wclk,
    input sclk,
    input rst_n,
    input ann_rdy, 
    input init_features_end,
    input mode,
    input [ACTIVATION_BUF_LEN1-1:0] act_sr1,
    input [ACTIVATION_BUF_LEN3-1:0] act_sr3,
    input [ACTIVATION_BUF_LEN4-1:0] act_sr4,
    input [SRAM1024_AW-1:0] addr_ann1_w_init,
    input [SRAM1024_AW-1:0] addr_ann1_b_init,
    input [SRAM1024_AW-1:0] addr_ann2_w_init,
    input [SRAM1024_AW-1:0] addr_ann2_b_init,
    input [SRAM8192_AW-1:0] addr_ann1_1_w_init,
    input [SRAM1024_AW-1:0] addr_ann1_1_b_init,
    input [SRAM8192_AW-1:0] addr_ann2_1_w_init,
    input [SRAM1024_AW-1:0] addr_ann2_1_b_init,
    input [SRAM8192_AW-1:0] addr_ann3_1_w_init,
    input [SRAM1024_AW-1:0] addr_ann3_1_b_init,
    input [SRAM8192_AW-1:0] addr_ann4_1_w_init,
    input [SRAM1024_AW-1:0] addr_ann4_1_b_init,
    input [SRAM8192_AW-1:0] addr_ann5_1_w_init,
    input [SRAM1024_AW-1:0] addr_ann5_1_b_init,
    input [SRAM8192_AW-1:0] addr_ann6_1_w_init,
    input [SRAM1024_AW-1:0] addr_ann6_1_b_init,
    input [SRAM8192_AW-1:0] addr_ann7_1_w_init,
    input [SRAM1024_AW-1:0] addr_ann7_1_b_init,
    input [SRAM8192_AW-1:0] addr_ann8_1_w_init,
    input [SRAM1024_AW-1:0] addr_ann8_1_b_init,
    input [SRAM8192_AW-1:0] addr_ann9_1_w_init,
    input [SRAM1024_AW-1:0] addr_ann9_1_b_init,
    input [SRAM8192_AW-1:0] addr_ann10_1_w_init,
    input [SRAM1024_AW-1:0] addr_ann10_1_b_init,
    input [SRAM8192_AW-1:0] addr_ann11_1_w_init,
    input [SRAM1024_AW-1:0] addr_ann11_1_b_init,
    input [SRAM8192_AW-1:0] addr_ann12_1_w_init,
    input [SRAM1024_AW-1:0] addr_ann12_1_b_init,
    input [SRAM1024_AW-1:0] addr_ann1_2_w_init,
    input [SRAM512_AW-1:0] addr_ann1_2_b_init,
    input [SRAM1024_AW-1:0] addr_ann2_2_w_init,
    input [SRAM512_AW-1:0] addr_ann2_2_b_init,
    input [SRAM1024_AW-1:0] addr_ann3_2_w_init,
    input [SRAM512_AW-1:0] addr_ann3_2_b_init,
    input [SRAM1024_AW-1:0] addr_ann4_2_w_init,
    input [SRAM512_AW-1:0] addr_ann4_2_b_init,
    input [SRAM1024_AW-1:0] addr_ann5_2_w_init,
    input [SRAM512_AW-1:0] addr_ann5_2_b_init,
    input [SRAM1024_AW-1:0] addr_ann6_2_w_init,
    input [SRAM512_AW-1:0] addr_ann6_2_b_init,
    input [SRAM1024_AW-1:0] addr_ann7_2_w_init,
    input [SRAM512_AW-1:0] addr_ann7_2_b_init,
    input [SRAM1024_AW-1:0] addr_ann8_2_w_init,
    input [SRAM512_AW-1:0] addr_ann8_2_b_init,
    input [SRAM1024_AW-1:0] addr_ann9_2_w_init,
    input [SRAM512_AW-1:0] addr_ann9_2_b_init,
    input [SRAM1024_AW-1:0] addr_ann10_2_w_init,
    input [SRAM512_AW-1:0] addr_ann10_2_b_init,
    input [SRAM512_AW-1:0] addr_ann11_2_w_init,
    input [SRAM512_AW-1:0] addr_ann11_2_b_init,
    input [SRAM512_AW-1:0] addr_ann12_2_w_init,
    input [SRAM512_AW-1:0] addr_ann12_2_b_init,
    input signed [SRAM16_DW-1:0] sram7_dout,
    input signed [SRAM16_DW-1:0] sram8_dout,
    input signed [SRAM16_DW-1:0] sram9_dout,
    input signed [SRAM16_DW-1:0] sram10_dout,
    input signed [SRAM16_DW-1:0] sram11_dout,
    input signed [SRAM16_DW-1:0] sram12_dout,
    input [PARAM_DW-1:0]  LEAD_THRES,
    output reg [4 : 0] ann_state,
    output sram7_en,
    output sram8_en,//
    output sram9_en,//
    output sram10_en,//
    output sram11_en,//
    output sram12_en,//

    output reg [SRAM8192_AW-1:0] addr_ann, 
    output reg signed [SRAM16_DW-1:0]  ann_w,
    output reg signed [SRAM16_DW-1:0]  ann_b,
    
    output [1:0] ann_shift,
    output feature_shift,
    output input_init_en,
    output ann_mi_1,
    output ann_mi_2,
    output spad_w_we_en,
    output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_re, 
    output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_we, 
    output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_a_addr_re, 
    output  [INPUT_DW*SPAD_DEPTH -1 : 0] spad_a_data_in, 
    output reg [1:0] mult_a_crl,
    output reg ann_out_vld,
    output reg ann_hidden_out_vld,
    input signed [INTEVAL_DW-1: 0] rr_diff,
    input signed [INTEVAL_DW -1 : 0] rr_pre_rr_ave,
    input signed [INTEVAL_DW -1 : 0] rr_post_rr_ave,
    input signed [INTEVAL_DW -1 : 0] qrs_cur_qrs_ave,
    input signed [INPUT_DW - 1: 0] r_amp_r_amp_ave,
    input signed [INPUT_DW - 1: 0] q_amp_q_amp_ave,
    input signed [INPUT_DW - 1: 0] s_amp_s_amp_ave,
    input signed [INPUT_DW - 1: 0] p_amp_p_amp_ave,
    input signed [INPUT_DW - 1: 0] t_amp_t_amp_ave,
    input signed [DIR_DW-1:0] t_dir,

    output [FEATURE_DIM*INPUT_DW-1:0] feature_matrix,
    output [2*NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW + DIR_DW -1:0] feature_matrix_mi,
    output signed [FEATURE_SUM_DW  -1:0] ann_mi_in,
    output signed [SRAM16_DW +INPUT_DW  -1:0] ann_hidden_in,
    output signed [FEATURE_SUM_DW +SRAM16_DW  -1:0] ann_mi_hidden_in,
    output ann_relu_en,
    output ann_done,
    input signed [INPUT_DW +SRAM16_DW-1:0 ] ann_out,
    input signed [SRAM16_DW + FEATURE_SUM_DW -1:0] ann_out_mi,
    output reg [ARR_LABEL_DW-1:0] arr_type,
    output reg mi_type);

    localparam N       = 5;

    
    reg [N-1 : 0] ann_state_next;

    localparam idle    = 5'd0;
    localparam ann1 = 5'd1; // the first layer of arr ann
    localparam ann2 = 5'd2; // the second layer of arr ann
    localparam done    = 5'd3; 
    localparam ann1_1 = 5'd4;// the first layer of mi ann, lead ii
    localparam ann1_2 = 5'd5;
    localparam ann2_1 = 5'd6;
    localparam ann2_2 = 5'd7;
    localparam ann3_1 = 5'd8;
    localparam ann3_2 = 5'd9;
    localparam ann4_1 = 5'd10;
    localparam ann4_2 = 5'd11;
    localparam ann5_1 = 5'd12;
    localparam ann5_2 = 5'd13;
    localparam ann6_1 = 5'd14;
    localparam ann6_2 = 5'd15;
    localparam ann7_1 = 5'd16;
    localparam ann7_2 = 5'd17;
    localparam ann8_1 = 5'd18;
    localparam ann8_2 = 5'd19;
    localparam ann9_1 = 5'd20;
    localparam ann9_2 = 5'd21;
    localparam ann10_1 = 5'd22;
    localparam ann10_2 = 5'd23;
    localparam ann11_1 = 5'd24;
    localparam ann11_2 = 5'd25;
    localparam ann12_1 = 5'd26;
    localparam ann12_2 = 5'd27;

    wire layer_done;

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            ann_state <= idle;
        else
            ann_state <= ann_state_next;
    end


    always @(*) begin
        case (ann_state)
            idle: begin
                if (ann_rdy) begin
                    if (mode == 0)
                        ann_state_next = ann1;
                    else begin
                        if (init_features_end) ann_state_next = ann1;
                        else ann_state_next = ann1_1;
                    end
                end
                else
                    ann_state_next = idle;
            end
            ann1: begin
                if (layer_done)
                    ann_state_next = ann2;
                else ann_state_next = ann1;
            end
            ann2: begin
                if (layer_done) begin
                    if (mode == 0) ann_state_next = done;
                    else ann_state_next = ann1_1;
                end  
                else ann_state_next = ann2;
            end
            ann1_1: begin
                if (layer_done)
                    ann_state_next = ann1_2;
                else ann_state_next = ann1_1;                
            end
            ann1_2:begin
                if (layer_done)
                    ann_state_next = ann2_1;
                else ann_state_next = ann1_2;                
            end
            ann2_1: begin
                if (layer_done)
                    ann_state_next = ann2_2;
                else ann_state_next = ann2_1;                 
            end
            ann2_2:begin
                if (layer_done)
                    ann_state_next = ann3_1;
                else ann_state_next = ann2_2;                 
            end
            ann3_1: begin
                if (layer_done)
                    ann_state_next = ann3_2;
                else ann_state_next = ann3_1;                 
            end
            ann3_2:begin
                if (layer_done)
                    ann_state_next = ann4_1;
                else ann_state_next = ann3_2;                 
            end  
            ann4_1: begin
                if (layer_done)
                    ann_state_next = ann4_2;
                else ann_state_next = ann4_1;                 
            end
            ann4_2:begin
                if (layer_done)
                    ann_state_next = ann5_1;
                else ann_state_next = ann4_2;                 
            end
            ann5_1: begin
                if (layer_done)
                    ann_state_next = ann5_2;
                else ann_state_next = ann5_1;                 
            end
            ann5_2:begin
                if (layer_done)
                    ann_state_next = ann6_1;
                else ann_state_next = ann5_2;                 
            end 
            ann6_1: begin
                if (layer_done)
                    ann_state_next = ann6_2;
                else ann_state_next = ann6_1;                 
            end
            ann6_2:begin
                if (layer_done)
                    ann_state_next = ann7_1;
                else ann_state_next = ann6_2;                 
            end   
            ann7_1: begin
                if (layer_done)
                    ann_state_next = ann7_2;
                else ann_state_next = ann7_1;                 
            end
            ann7_2:begin
                if (layer_done)
                    ann_state_next = ann8_1;
                else ann_state_next = ann7_2;                 
            end
            ann8_1: begin
                if (layer_done)
                    ann_state_next = ann8_2;
                else ann_state_next = ann8_1;                 
            end
            ann8_2:begin
                if (layer_done)
                    ann_state_next = ann9_1;
                else ann_state_next = ann8_2;                 
            end  
            ann9_1: begin
                if (layer_done)
                    ann_state_next = ann9_2;
                else ann_state_next = ann9_1;                 
            end
            ann9_2:begin
                if (layer_done)
                    ann_state_next = ann10_1;
                else ann_state_next = ann9_2;                 
            end 
            ann10_1: begin
                if (layer_done)
                    ann_state_next = ann10_2;
                else ann_state_next = ann10_1;                
            end
            ann10_2:begin
                if (layer_done)
                    ann_state_next = ann11_1;
                else ann_state_next = ann10_2;                
            end
            ann11_1: begin
                if (layer_done)
                    ann_state_next = ann11_2;
                else ann_state_next = ann11_1;                
            end
            ann11_2:begin
                if (layer_done)
                    ann_state_next = ann12_1;
                else ann_state_next = ann11_2;                
            end
            ann12_1: begin
                if (layer_done)
                    ann_state_next = ann12_2;
                else ann_state_next = ann12_1;                
            end
            ann12_2:begin
                if (layer_done)
                    ann_state_next = done;
                else ann_state_next = ann12_2;                
            end              
            done: begin
                    ann_state_next = idle;
            end
            default:ann_state_next = idle;
        endcase
    end
    assign ann_mi_1 = (ann_state == ann1_1)|
                      (ann_state == ann2_1)|
                      (ann_state == ann3_1)|
                      (ann_state == ann4_1)|
                      (ann_state == ann5_1)|
                      (ann_state == ann6_1)|
                      (ann_state == ann7_1)|
                      (ann_state == ann8_1)|
                      (ann_state == ann9_1)|
                      (ann_state == ann10_1)|
                      (ann_state == ann11_1)|
                      (ann_state == ann12_1);

    assign ann_mi_2 = (ann_state == ann1_2)|
                      (ann_state == ann2_2)|
                      (ann_state == ann3_2)|
                      (ann_state == ann4_2)|
                      (ann_state == ann5_2)|
                      (ann_state == ann6_2)|
                      (ann_state == ann7_2)|
                      (ann_state == ann8_2)|
                      (ann_state == ann9_2)|
                      (ann_state == ann10_2)|
                      (ann_state == ann11_2)|
                      (ann_state == ann12_2);

    assign ann_done = (ann_state == done)? 1:0;
    reg layer_done_d;
    reg ann_rdy_d;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            layer_done_d <= 0;
            ann_rdy_d <= 0;
            end
        else begin
            layer_done_d <= layer_done;
            ann_rdy_d <= ann_rdy;
        end
           
    end    
    wire layer_rdy;
    assign layer_rdy = ann_rdy_d | (layer_done_d & (ann_state != done));     
    

    reg [SRAM8192_AW-1:0] addr_ann_w_init;    // from top.v,
    reg [SRAM8192_AW-1:0] addr_ann_b_init;    // from top.v,

    always @(*) begin
        case(ann_state_next)
        idle: begin
            addr_ann_w_init = {3'b0,addr_ann1_w_init};
            addr_ann_b_init = {3'b0,addr_ann1_b_init};
        end
        ann1: begin
            addr_ann_w_init = {3'b0,addr_ann1_w_init};
            addr_ann_b_init = {3'b0,addr_ann1_b_init};    
        end
        ann2:begin
            addr_ann_w_init = {3'b0,addr_ann2_w_init};
            addr_ann_b_init = {3'b0,addr_ann2_b_init};     
        end
        ann1_1:begin
            addr_ann_w_init = addr_ann1_1_w_init;
            addr_ann_b_init = {3'b0,addr_ann1_1_b_init};
        end
        ann1_2:begin
            addr_ann_w_init = {3'b0,addr_ann1_2_w_init};
            addr_ann_b_init = {5'b0,addr_ann1_2_b_init};
        end
        ann2_1:begin
            addr_ann_w_init = addr_ann2_1_w_init;
            addr_ann_b_init = {3'b0,addr_ann2_1_b_init};
        end
        ann2_2:begin
            addr_ann_w_init = {3'b0,addr_ann2_2_w_init};
            addr_ann_b_init = {5'b0,addr_ann2_2_b_init};
        end
        ann3_1:begin
            addr_ann_w_init = addr_ann3_1_w_init;
            addr_ann_b_init = {3'b0,addr_ann3_1_b_init};
        end
        ann3_2:begin
            addr_ann_w_init = {3'b0,addr_ann3_2_w_init};
            addr_ann_b_init = {5'b0,addr_ann3_2_b_init};
        end
        ann4_1:begin
            addr_ann_w_init = addr_ann4_1_w_init;
            addr_ann_b_init = {3'b0,addr_ann4_1_b_init};
        end
        ann4_2:begin
            addr_ann_w_init = {3'b0,addr_ann4_2_w_init};
            addr_ann_b_init = {5'b0,addr_ann4_2_b_init};
        end
        ann5_1:begin
            addr_ann_w_init = addr_ann5_1_w_init;
            addr_ann_b_init = {3'b0,addr_ann5_1_b_init};
        end
        ann5_2:begin
            addr_ann_w_init = {3'b0,addr_ann5_2_w_init};
            addr_ann_b_init = {5'b0,addr_ann5_2_b_init};
        end
        ann6_1:begin
            addr_ann_w_init = addr_ann6_1_w_init;
            addr_ann_b_init = {3'b0,addr_ann6_1_b_init};
        end
        ann6_2:begin
            addr_ann_w_init = {3'b0,addr_ann6_2_w_init};
            addr_ann_b_init = {5'b0,addr_ann6_2_b_init};
        end
        ann7_1:begin
            addr_ann_w_init = addr_ann7_1_w_init;
            addr_ann_b_init = {3'b0,addr_ann7_1_b_init};
        end
        ann7_2:begin
            addr_ann_w_init = {3'b0,addr_ann7_2_w_init};
            addr_ann_b_init = {5'b0,addr_ann7_2_b_init};
        end
        ann8_1:begin
            addr_ann_w_init = addr_ann8_1_w_init;
            addr_ann_b_init = {3'b0,addr_ann8_1_b_init};
        end
        ann8_2:begin
            addr_ann_w_init = {3'b0,addr_ann8_2_w_init};
            addr_ann_b_init = {5'b0,addr_ann8_2_b_init};
        end
        ann9_1:begin
            addr_ann_w_init = addr_ann9_1_w_init;
            addr_ann_b_init = {3'b0,addr_ann9_1_b_init};
        end
        ann9_2:begin
            addr_ann_w_init = {3'b0,addr_ann9_2_w_init};
            addr_ann_b_init = {5'b0,addr_ann9_2_b_init};
        end
        ann10_1:begin
            addr_ann_w_init = addr_ann10_1_w_init;
            addr_ann_b_init = {3'b0,addr_ann10_1_b_init};
        end
        ann10_2:begin
            addr_ann_w_init = {3'b0,addr_ann10_2_w_init};
            addr_ann_b_init = {5'b0,addr_ann10_2_b_init};
        end
        ann11_1:begin
            addr_ann_w_init = addr_ann11_1_w_init;
            addr_ann_b_init = {3'b0,addr_ann11_1_b_init};
        end
        ann11_2:begin
            addr_ann_w_init = {5'b0,addr_ann11_2_w_init};
            addr_ann_b_init = {5'b0,addr_ann11_2_b_init};
        end
        ann12_1:begin
            addr_ann_w_init = addr_ann12_1_w_init;
            addr_ann_b_init = {3'b0,addr_ann12_1_b_init};
        end
        ann12_2:begin
            addr_ann_w_init = {5'b0,addr_ann12_2_w_init};
            addr_ann_b_init = {5'b0,addr_ann12_2_b_init};
        end
        default:begin
            addr_ann_w_init = {3'b0,addr_ann1_w_init};
            addr_ann_b_init = {3'b0,addr_ann1_b_init};
        end 
        endcase
    end

    localparam prepare_input = 3'b100;
    localparam load_w = 3'b001;
    localparam load_b = 3'b110;
    localparam load_a  = 3'b010;
    localparam mac     = 3'b011;
    localparam add_b   = 3'b101;
    localparam one_layer_done = 3'b111;
    reg [2 : 0] ann_layer_state;
    reg [2 : 0] ann_layer_state_next;
    // assign ann_w = ((ann_layer_state == load_w) & ((ann_state == ann1) |(ann_state == ann2))) ? sram7_dout :0;
    always @(*) begin
        if (ann_layer_state == load_w) begin
            if ((ann_state == ann1) |(ann_state == ann2)) ann_w = sram7_dout;
            else if ((ann_state == ann1_1)|(ann_state == ann2_1)|(ann_state == ann3_1)|(ann_state == ann4_1)) ann_w = sram8_dout;
            else if ((ann_state == ann5_1)|(ann_state == ann6_1)|(ann_state == ann7_1)|(ann_state == ann8_1)) ann_w = sram9_dout;
            else if ((ann_state == ann9_1)|(ann_state == ann10_1)|(ann_state == ann11_1)|(ann_state == ann12_1)) ann_w = sram10_dout;
            else if ((ann_state == ann1_2)|(ann_state == ann2_2)|(ann_state == ann3_2)|(ann_state == ann4_2) |
                    (ann_state == ann5_2)|(ann_state == ann6_2)|(ann_state == ann7_2)|(ann_state == ann8_2)|
                    (ann_state == ann9_2)|(ann_state == ann10_2)) ann_w = sram11_dout;
            else if ((ann_state == ann11_2) | (ann_state == ann12_2)) ann_w = sram12_dout;
            else  ann_w = 0;
        end
        else ann_w = 0;
    end


    // control counter
    localparam  ONEDIM_TIMES_1 = 3;
    localparam  ONEDIM_TIMES_1_MI = FEATURE_DIM_MI/SPAD_DEPTH;

    localparam  OUT_DIM_1 = ANN_HIDDEN_DIM;
    localparam  ONEDIM_TIMES_2 = ANN_HIDDEN_DIM/SPAD_DEPTH;

    localparam  OUT_DIM_2 = ANN_OUT_DIM;
    localparam  OUT_DIM_2_MI = ANN_OUT_DIM_MI;

    localparam  W_REMAIN = 6;
    

    reg [$clog2(SPAD_DEPTH+1)-1 : 0] cnt_ks;
    reg [$clog2(ONEDIM_TIMES_1_MI+1)-1 : 0] cnt_onedim;
    reg [$clog2(OUT_DIM_1+1)-1 : 0] cnt_dim;
    reg emb_end;
    reg emb_end_d;

    

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            ann_layer_state <= idle;
        else
            ann_layer_state <= ann_layer_state_next;
    end

    always @(*) begin
        case (ann_layer_state)
            idle: begin
                if (layer_rdy)
                    ann_layer_state_next = prepare_input;
                else
                    ann_layer_state_next = idle;
            end
            prepare_input: begin
                ann_layer_state_next = load_w;
            end
            load_w: begin
                ann_layer_state_next = load_a;
            end
            load_a: begin
                if ( ann_state == ann1) begin
                    if (cnt_onedim != ONEDIM_TIMES_1-1) ann_layer_state_next = mac;
                    else ann_layer_state_next = load_b;
                end
                else if (ann_state == ann2) begin
                    if (cnt_onedim != ONEDIM_TIMES_2-1) ann_layer_state_next = mac;
                    else ann_layer_state_next = load_b;                    
                end
                else if (ann_mi_1) begin
                    if (cnt_onedim != ONEDIM_TIMES_1_MI-1) ann_layer_state_next = mac;
                    else ann_layer_state_next = load_b;                        
                end
                else if (ann_mi_2) begin
                    if (cnt_onedim != ONEDIM_TIMES_2-1) ann_layer_state_next = mac;
                    else ann_layer_state_next = load_b;                       
                end
                else;

            end
            load_b:ann_layer_state_next = mac;
            mac: begin
                if ((ann_state == ann1) & (cnt_onedim == 0)) begin
                    if (cnt_ks != W_REMAIN)
                        ann_layer_state_next = mac;
                    else 
                        ann_layer_state_next = add_b;
                end
                else begin
                    if (cnt_ks != SPAD_DEPTH)
                        ann_layer_state_next = mac;
                    else begin
                        if ( ann_state == ann1) begin
                            ann_layer_state_next = load_w;
                        end
                        else if (ann_state == ann2 | ann_mi_1 | ann_mi_2) begin
                            if (cnt_onedim != 0) ann_layer_state_next = load_w;
                            else ann_layer_state_next = add_b;                        
                        end
                        else;
                    end                    
                end

            end
            add_b: begin
                if ( ann_state == ann1 | ann_mi_1) begin
                    if (cnt_dim == OUT_DIM_1-1)  ann_layer_state_next = one_layer_done;
                    else ann_layer_state_next = load_w;
                end
                else if (ann_state == ann2) begin
                    if (cnt_dim == OUT_DIM_2-1)  ann_layer_state_next = one_layer_done;
                    else ann_layer_state_next = load_w;                    
                end
                else if (ann_mi_2) begin
                    if (cnt_dim == OUT_DIM_2_MI-1)  ann_layer_state_next = one_layer_done;
                    else ann_layer_state_next = load_w;                     
                end
                else;
            end
            
            one_layer_done:
                ann_layer_state_next = idle;
            default:ann_layer_state_next = idle;
        endcase
    end    
    assign layer_done = (ann_layer_state == one_layer_done)? 1:0;

    // control counter

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            cnt_ks  <= 0;
            cnt_onedim  <= 0;
            cnt_dim <= 0;
        end
        else begin
            if (ann_layer_state == mac) begin 
                cnt_dim <= cnt_dim;    
                cnt_onedim <= cnt_onedim;
                if ((ann_state == ann1) & (cnt_onedim == 0)) begin
                    cnt_ks  <= (cnt_ks == W_REMAIN)? 0:cnt_ks+1;
                end
                else begin
                    cnt_ks  <= (cnt_ks == SPAD_DEPTH)? 0:cnt_ks+1;                
                end
            end
            else if (ann_layer_state == load_a) begin
                if(ann_state == ann1)
                    cnt_onedim <= (cnt_onedim == ONEDIM_TIMES_1-1)? 0 : cnt_onedim + 1'b1;
                else if ((ann_state == ann2 ) | ann_mi_2)
                    cnt_onedim <= (cnt_onedim == ONEDIM_TIMES_2-1)? 0 : cnt_onedim + 1'b1;
                else if (ann_mi_1)
                    cnt_onedim <= (cnt_onedim == ONEDIM_TIMES_1_MI-1)? 0 : cnt_onedim + 1'b1;
                else;                
            end
            else if (ann_layer_state == add_b)  begin
                if ((ann_state == ann1) | ann_mi_1) begin
                    cnt_dim <= (cnt_dim == OUT_DIM_1-1)? 0 : cnt_dim + 1'b1;
                    cnt_onedim <= cnt_onedim;
                    cnt_ks <= cnt_ks;
                end
                else if (ann_state == ann2) begin
                    cnt_dim <= (cnt_dim == OUT_DIM_2-1)? 0 : cnt_dim + 1'b1;
                    cnt_onedim <= cnt_onedim;
                    cnt_ks <= cnt_ks;                    
                end
                else if (ann_mi_2) begin
                    cnt_dim <= (cnt_dim == OUT_DIM_2_MI-1)? 0 : cnt_dim + 1'b1;
                    cnt_onedim <= cnt_onedim;
                    cnt_ks <= cnt_ks;                      
                end
                else;
            end
            else if (ann_layer_state == done) begin
                cnt_ks  <= 0;
                cnt_onedim  <= 0;
                cnt_dim <= 0;                
            end
            else begin
                cnt_ks  <= cnt_ks;
                cnt_dim  <= cnt_dim;
                cnt_onedim <= cnt_onedim;
            end
        end
    end



    assign input_init_en = ( (ann_layer_state == prepare_input) & ((ann_state == ann1)|ann_mi_1))? 1:0; 
    wire signed [INPUT_DW-1:0] rr_diff_ex;
    wire signed [INPUT_DW-1:0] rr_pre_rr_ave_ex;
    wire signed [INPUT_DW-1:0] rr_post_rr_ave_ex;
    wire signed [INPUT_DW-1:0] qrs_cur_qrs_ave_ex;
    wire signed [INPUT_DW-1:0] t_dir_ex;
    wire signed [DIR_DW-1:0] t_dir_real;
    assign t_dir_real = (mode == 0)? t_dir: act_sr3[NUM_FEAS_MI* FEATURE_SUM_DW+ DIR_DW-1-:DIR_DW]; // if mode == 0 t_dir is lead ii, else t_dir is the last lead

    assign rr_diff_ex = {{(INPUT_DW-INTEVAL_DW){rr_diff[INTEVAL_DW-1]}},rr_diff };
    assign rr_pre_rr_ave_ex = {{(INPUT_DW-INTEVAL_DW){rr_pre_rr_ave[INTEVAL_DW-1]}},rr_pre_rr_ave };
    assign rr_post_rr_ave_ex = {{(INPUT_DW-INTEVAL_DW){rr_post_rr_ave[INTEVAL_DW-1]}},rr_post_rr_ave };
    assign qrs_cur_qrs_ave_ex = {{(INPUT_DW-INTEVAL_DW){qrs_cur_qrs_ave[INTEVAL_DW-1]}},qrs_cur_qrs_ave };
    assign t_dir_ex = {{(INPUT_DW-2){t_dir_real[DIR_DW-1]}},t_dir_real };


    assign feature_matrix = {t_dir_ex,t_dir_ex,
                            t_amp_t_amp_ave, p_amp_p_amp_ave,
                            s_amp_s_amp_ave, q_amp_q_amp_ave,
                            r_amp_r_amp_ave, r_amp_r_amp_ave,
                            qrs_cur_qrs_ave_ex, qrs_cur_qrs_ave_ex,
                            rr_post_rr_ave_ex, rr_pre_rr_ave_ex, rr_diff_ex,
                            rr_post_rr_ave_ex, rr_pre_rr_ave_ex, rr_diff_ex,
                            rr_post_rr_ave_ex, rr_pre_rr_ave_ex, rr_diff_ex,
                            rr_post_rr_ave_ex, rr_pre_rr_ave_ex, rr_diff_ex};

    // wire [FEATURE_SUM_DW-1:0] fm_st_slo_sum;
    // wire [FEATURE_SUM_DW-1:0] fm_st_amp_iso_sum;
    // wire [FEATURE_SUM_DW-1:0] fm_r_amp_iso_sum;
    // wire [FEATURE_SUM_DW-1:0] fm_t_amp_iso_sum;
    // wire [FEATURE_SUM_DW-1:0] fm_s_amp_iso_sum;
    // wire [FEATURE_SUM_DW-1:0] fm_q_amp_iso_sum ;
    // wire [FEATURE_SUM_DW-1:0] fm_r_amp_t_amp_sum;
    // wire [QRS_EMB_LEN*EMB_DW-1:0] fm_qrs_emb_buffer;
    // wire [T_EMB_LEN*EMB_DW-1:0] fm_t_emb_buffer;
    // wire [DIR_DW-1:0] fm_t_dir;
    // wire [FEATURE_SUM_DW-1:0] fm_st_slo;
    // wire [FEATURE_SUM_DW-1:0] fm_st_amp_iso;
    // wire [FEATURE_SUM_DW-1:0] fm_r_amp_iso;
    // wire [FEATURE_SUM_DW-1:0] fm_t_amp_iso;
    // wire [FEATURE_SUM_DW-1:0] fm_s_amp_iso;
    // wire [FEATURE_SUM_DW-1:0] fm_q_amp_iso;
    // wire [FEATURE_SUM_DW-1:0] fm_r_amp_t_amp;
    
    // assign fm_st_slo_sum = feature_matrix_mi[2*NUM_FEAS_MI * FEATURE_SUM_DW + DIR_DW  +   (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW-1-: FEATURE_SUM_DW];
    // assign fm_st_amp_iso_sum = feature_matrix_mi[(2*NUM_FEAS_MI -1) * FEATURE_SUM_DW + DIR_DW  +   (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW-1-: FEATURE_SUM_DW];
    // assign fm_r_amp_iso_sum = feature_matrix_mi[(2*NUM_FEAS_MI -2) * FEATURE_SUM_DW + DIR_DW  +   (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW-1-: FEATURE_SUM_DW];
    // assign fm_t_amp_iso_sum = feature_matrix_mi[(2*NUM_FEAS_MI -3) * FEATURE_SUM_DW + DIR_DW  +   (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW-1-: FEATURE_SUM_DW];
    // assign fm_s_amp_iso_sum = feature_matrix_mi[(2*NUM_FEAS_MI -4) * FEATURE_SUM_DW + DIR_DW  +   (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW-1-: FEATURE_SUM_DW];
    // assign fm_q_amp_iso_sum  =feature_matrix_mi[(2*NUM_FEAS_MI -5) * FEATURE_SUM_DW + DIR_DW  +   (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW-1-: FEATURE_SUM_DW];
    // assign fm_r_amp_t_amp_sum = feature_matrix_mi[(2*NUM_FEAS_MI -6) * FEATURE_SUM_DW + DIR_DW  +   (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW-1-: FEATURE_SUM_DW];
    // assign fm_qrs_emb_buffer =feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + DIR_DW  +   (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW-1-: QRS_EMB_LEN *EMB_DW];
    // assign fm_t_emb_buffer = feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + DIR_DW  +   ( T_EMB_LEN) * EMB_DW-1-: T_EMB_LEN * EMB_DW];
    // assign fm_t_dir= feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW + DIR_DW  -1-: DIR_DW];
    // assign fm_st_slo = feature_matrix_mi[NUM_FEAS_MI * FEATURE_SUM_DW   -1-: FEATURE_SUM_DW];
    // assign fm_st_amp_iso = feature_matrix_mi[(NUM_FEAS_MI-1) * FEATURE_SUM_DW   -1-: FEATURE_SUM_DW];
    // assign fm_r_amp_iso = feature_matrix_mi[(NUM_FEAS_MI-2) * FEATURE_SUM_DW   -1-: FEATURE_SUM_DW];
    // assign fm_t_amp_iso = feature_matrix_mi[(NUM_FEAS_MI-3) * FEATURE_SUM_DW   -1-: FEATURE_SUM_DW];
    // assign fm_s_amp_iso = feature_matrix_mi[(NUM_FEAS_MI-4) * FEATURE_SUM_DW   -1-: FEATURE_SUM_DW];
    // assign fm_q_amp_iso = feature_matrix_mi[(NUM_FEAS_MI-5) * FEATURE_SUM_DW   -1-: FEATURE_SUM_DW];
    // assign fm_r_amp_t_amp = feature_matrix_mi[(NUM_FEAS_MI-6) * FEATURE_SUM_DW   -1-: FEATURE_SUM_DW];

    assign feature_matrix_mi =  {act_sr3[NUM_FEAS_MI * (NUM_LEADS+1) * FEATURE_SUM_DW + DIR_DW * NUM_LEADS + NUM_LEADS * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW-1-:NUM_FEAS_MI*FEATURE_SUM_DW],
                                act_sr3[NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW + DIR_DW-1:0]};
    assign feature_shift = ((mode == 1) & ann_mi_1 & input_init_en)? 1:0;

    reg [SRAM8192_AW-1:0] addr_ann_w; 
    reg [SRAM8192_AW-1:0] addr_ann_b;
    reg spad_w_addr_we_end;
    reg addr_ann_b_end;
    always @(negedge sclk or negedge rst_n) begin
        if (!rst_n) begin
            addr_ann_w <= 0;
            spad_w_addr_we <= 0;
            spad_w_addr_we_end <= 0;
        end
        else begin
            if ((ann_layer_state == load_w) & (!spad_w_addr_we_end)) begin
                if ((ann_state == ann1) & (cnt_onedim ==  ONEDIM_TIMES_1-1)) begin
                    if (spad_w_addr_we == W_REMAIN-1) begin
                        spad_w_addr_we <= 0;
                        spad_w_addr_we_end <= 1;
                        addr_ann_w <= addr_ann_w + 1;
                    end
                    else begin
                        addr_ann_w <= addr_ann_w + 1;
                        spad_w_addr_we <= spad_w_addr_we + 1; 
                        spad_w_addr_we_end <= spad_w_addr_we_end;                   
                    end                    
                end
                else begin
                    if (spad_w_addr_we == SPAD_DEPTH-1) begin
                        spad_w_addr_we <= 0;
                        spad_w_addr_we_end <= 1;
                        addr_ann_w <= addr_ann_w+ 1;
                    end
                    else begin
                        addr_ann_w <= addr_ann_w + 1;
                        spad_w_addr_we <= spad_w_addr_we + 1; 
                        spad_w_addr_we_end <= spad_w_addr_we_end;                   
                    end
                end

            end
            else if (ann_layer_state == load_a) begin
                    spad_w_addr_we <= 0;    // rst
                    spad_w_addr_we_end <= 0; // rst
                    addr_ann_w <= addr_ann_w ;                
            end
            else if (ann_layer_state == idle) begin //reset
                addr_ann_w <= addr_ann_w_init;
                spad_w_addr_we <= 0;       
                spad_w_addr_we_end <= 0; // rst         
            end
            else begin
                addr_ann_w <= addr_ann_w ;
                spad_w_addr_we <= spad_w_addr_we;  
                spad_w_addr_we_end <= spad_w_addr_we_end ;              
            end
        end
    end
    assign spad_w_we_en = ((ann_layer_state == load_w) & (!spad_w_addr_we_end)) ? 1:0;

    always @(negedge sclk or negedge rst_n) begin
        if (!rst_n) begin
            addr_ann_b <= 0;
            addr_ann_b_end <= 0;
        end
        else begin
            if (ann_layer_state == load_b) begin
                if (!addr_ann_b_end) begin
                    addr_ann_b <= addr_ann_b + 1;
                    addr_ann_b_end <= 1;
                end
                else begin
                    addr_ann_b <= addr_ann_b; 
                    addr_ann_b_end <= addr_ann_b_end;
                end
            end
            else if (ann_layer_state ==  add_b) begin
                addr_ann_b_end <= 0;
                addr_ann_b <= addr_ann_b; 
            end
            else if (ann_layer_state == idle) begin
                addr_ann_b <= addr_ann_b_init ;
                addr_ann_b_end <= 0;       
            end
            else begin
                addr_ann_b <= addr_ann_b;
                addr_ann_b_end <= addr_ann_b_end;
            end
        end
    end

    always  @(negedge sclk or negedge rst_n) begin
        if (!rst_n)
            ann_b <= 0;
        else begin
            if ((ann_layer_state == load_b) & (!addr_ann_b_end)) begin
                if ((ann_state == ann1)|(ann_state == ann2))
                    ann_b <= sram7_dout;
                else if (ann_mi_1) ann_b <= sram11_dout;
                else if (ann_mi_2) ann_b <= sram12_dout;
                else ann_b <= ann_b;
            end
            
            else
                ann_b <= ann_b;
        end
    end

    always @(*) begin
        if (ann_layer_state == load_w) addr_ann = addr_ann_w;
        else if (ann_layer_state == load_b) addr_ann = addr_ann_b;
        else addr_ann = 0;        
    end 
    assign sram7_en  = ((ann_state == ann1)|(ann_state == ann2))?(((ann_layer_state == load_w) | (ann_layer_state == load_b)) ? 1:0):0; // scale and b are on sram 7
    assign sram8_en  = ((ann_state == ann1_1)|(ann_state == ann2_1)|(ann_state == ann3_1)|(ann_state == ann4_1))?((ann_layer_state == load_w) ? 1:0):0; // ann1_1_w,ann1_2_w,ann2_1_w,ann2_2_w are on sram 8
    assign sram9_en  = ((ann_state == ann5_1)|(ann_state == ann6_1)|(ann_state == ann7_1)|(ann_state == ann8_1))?((ann_layer_state == load_w) ? 1:0):0;
    assign sram10_en  = ((ann_state == ann9_1)|(ann_state == ann10_1)|(ann_state == ann11_1)|(ann_state == ann12_1))?((ann_layer_state == load_w) ? 1:0):0;
    assign sram11_en  = ((ann_mi_1 & (ann_layer_state == load_b) )| (ann_mi_2 & !(ann_state == ann11_2) & !(ann_state == ann12_2) & (ann_layer_state == load_w)))? 1:0;
    assign sram12_en  = ((ann_mi_2 & (ann_layer_state == load_b))|(((ann_state == ann11_2) | (ann_state == ann12_2)) & (ann_layer_state == load_w))) ? 1:0;

    //load_a
    assign ann_shift       = (ann_state == ann1)? ((ann_layer_state == load_a)? 2'b01:0): (((ann_layer_state == mac) & (cnt_ks < SPAD_DEPTH))?2'b10:0);

    assign spad_a_data_in = (ann_state ==  ann1) ? ((ann_layer_state == load_a)?act_sr4[SPAD_DEPTH*INPUT_DW-1:0]:0):0;
    assign ann_mi_in =  (ann_mi_1) ? ((ann_layer_state == mac)?act_sr4[FEATURE_SUM_DW-1:0]:0):0;
    assign ann_hidden_in =  (ann_state ==  ann2) ? ((ann_layer_state == mac)?act_sr1[INPUT_DW+ SRAM16_DW-1:0]:0):0;
    assign ann_mi_hidden_in = (ann_mi_2) ? ((ann_layer_state == mac)?act_sr1[FEATURE_SUM_DW+ SRAM16_DW-1:0]:0):0;

    //mac
        
    always@(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            spad_w_addr_re <= 0 ;
            spad_a_addr_re <= 0;
        end
        else begin
            if (ann_layer_state == mac) begin
                if ((ann_state == ann1) & (cnt_onedim == 0)) begin
                    spad_w_addr_re <= (spad_w_addr_re == W_REMAIN-1)? spad_w_addr_re: spad_w_addr_re + 1;
                    spad_a_addr_re <= (spad_a_addr_re == W_REMAIN-1)? spad_a_addr_re : spad_a_addr_re + 1;                    
                end
                else begin
                    spad_w_addr_re <= (spad_w_addr_re == SPAD_DEPTH-1)? spad_w_addr_re: spad_w_addr_re + 1;
                    spad_a_addr_re <= (spad_a_addr_re == SPAD_DEPTH-1)? spad_a_addr_re : spad_a_addr_re + 1;
                end
            end
            else begin
                spad_w_addr_re <= 0;
                spad_a_addr_re <= 0;
            end
        end
    end
    reg add_b_d;
    always@(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            add_b_d <= 0 ;
        end
        else begin
            add_b_d <= (ann_layer_state == add_b)? 1:0;
        end
    end
    always @(*) begin
        if (ann_layer_state == mac) begin
            mult_a_crl = 2'b01;
        end
        else if (ann_layer_state == add_b) begin
            mult_a_crl = 2'b11;
        end
        else if (add_b_d) begin
            mult_a_crl = 2'b00;
        end 
        else begin
            mult_a_crl = 2'b10;
        end
    end
    always@(posedge wclk or negedge rst_n)
    begin
        if (!rst_n) begin
            ann_out_vld <= 0;
            ann_hidden_out_vld <= 0;
        end
        else
        begin
            if ((ann_state == ann1) | ann_mi_1) begin
                if (mult_a_crl == 2'b11) ann_hidden_out_vld <= 1;
                else ann_hidden_out_vld <= 0;                

            end
            else if ((ann_state == ann2)|ann_mi_2) begin
                if (mult_a_crl == 2'b11) ann_out_vld <= 1;
                else ann_out_vld <= 0;
            end
            else;
        end
    end    
    assign ann_relu_en = (ann_out_vld) | (ann_hidden_out_vld);

    reg signed [SRAM16_DW +INPUT_DW  -1:0] ann_out_temp;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            arr_type <= 0;
            ann_out_temp <= {1'b1,{(SRAM16_DW + INPUT_DW - 1 ){1'b0}}};
        end
        else begin
            if (ann_state == ann2) begin
                if (ann_out_vld) begin
                    if (ann_out > ann_out_temp) begin
                        arr_type <= cnt_dim;
                        ann_out_temp <= ann_out;

                    end
                    else begin
                        arr_type <= arr_type;
                        ann_out_temp <= ann_out_temp;
                    end
                end
            end
            else if (ann_state == done) begin //rst
                arr_type <= 0;
                ann_out_temp <= {1'b1,{(SRAM16_DW + INPUT_DW - 1 ){1'b0}}};           
            end
        end
    end
    reg signed [SRAM16_DW +FEATURE_SUM_DW  -1:0] ann_out_mi_temp;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            ann_out_mi_temp <= {1'b1,{(SRAM16_DW +FEATURE_SUM_DW - 1 ){1'b0}}};
        end
        else begin
            if (ann_mi_2) begin
                if (ann_out_vld) begin
                    if (ann_out_mi > ann_out_mi_temp) begin
                        ann_out_mi_temp <= ann_out_mi;
                    end
                    else begin
                        ann_out_mi_temp <= ann_out_mi_temp;
                    end
                end
                else if (ann_layer_state == idle) begin//before the next valid ann_out_vld rst
                    ann_out_mi_temp <= {1'b1,{(SRAM16_DW +FEATURE_SUM_DW - 1 ){1'b0}}};                   
                end
            end
            else if (ann_state == done) begin 
                ann_out_mi_temp <= {1'b1,{(SRAM16_DW +FEATURE_SUM_DW - 1 ){1'b0}}};                 
            end
        end
    end
    reg [$clog2(NUM_LEADS+1)-1:0] sum_mi_lead;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            sum_mi_lead <= 0;
        end
        else begin
            if (ann_mi_2) begin
                if (ann_layer_state == one_layer_done) begin
                    if ((ann_out_mi > ann_out_mi_temp) & (ann_out_mi_temp != {1'b1,{(SRAM16_DW +FEATURE_SUM_DW - 1 ){1'b0}}})) begin
                        sum_mi_lead <= sum_mi_lead + 1;
                    end
                    else sum_mi_lead <= sum_mi_lead;
                end
                else if (ann_state == done) sum_mi_lead <= 0; //rst
            end
        end
    end 

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            mi_type <= 0;
        end
        else begin
            if (ann_state == done) begin
                if (sum_mi_lead > LEAD_THRES)
                    mi_type <= 1;
                else mi_type <= 0;
                
            end
            else begin
                mi_type <= mi_type;
            end
        end
    end
endmodule