`timescale  1ns/100ps
module DECODER #(parameter DATA_DW = 8,
    INPUT_DW = 12,
    DATA_BQ_DW = 32,
    W_DW = 8,
    B_DW = 32,
    SCALE_DW = 32,
    SRAM1024_AW = 10,
    SRAM8192_AW = 13,
    SRAM512_AW = 9,
    SRAM8_DW = 8,
    SRAM32_DW = 32, 
    SRAM16_DW = 16,
    SPAD_DEPTH = 8,  
    PE_NUM = 32, 
    DCNN1_CHIN = 64,
    DCNN1_LENGTH_IN = 64,
    DCNN1_CHOUT = 32,
    DCNN_STRIDE = 2,
    DCNN_PADDING = 3,
    DCNN_KS = 8,
    DCNN1_LENGTH_OUT = DCNN_STRIDE * (DCNN1_LENGTH_IN - 1) -  2 * DCNN_PADDING + DCNN_KS,
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
    CNN21_LENGTH_OUT =  DCNN2_LENGTH_OUT,
    CNN22_LENGTH_IN = DCNN2_LENGTH_OUT,
    CNN22_CHIN = CNN21_CHOUT,
    CNN22_CHOUT = 4,
    CNN22_LENGTH_OUT =  DCNN2_LENGTH_OUT,
    ADDR_LSTM10_SRAM_ACT_INIT = 6144,
    ADDR_LSTM11_SRAM_ACT_INIT = 0)
    (input wclk,
     input sclk,
     input rst_n,

    //  input [ACTIVATION_BUF_LEN2*DATA_DW-1:0] act_sr2, //from segment.v, XT_0, HT_0_REVERSE
     input [DATA_BQ_DW-1:0] act_sr1_1, // from segment.v , HT_0
    //  input [ACTIVATION_BUF_LEN3*DATA_DW-1:0] act_sr3,     
    //  input [ACTIVATION_BUF_LEN4*DATA_DW-1:0] act_sr4,
     input [SRAM8192_AW-1:0] addr_dcnn1_w_init,    // from top.v, w0
     input [SRAM1024_AW-1:0] addr_dcnn1_scales_init,  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale
     input [SRAM8192_AW-1:0] addr_cnn11_w_init,    // from top.v, w0
     input [SRAM1024_AW-1:0] addr_cnn11_scales_init,  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale
     input [SRAM1024_AW-1:0] addr_cnn11_b_init,
     input [SRAM8192_AW-1:0] addr_cnn12_w_init,    // from top.v, w0
     input [SRAM1024_AW-1:0] addr_cnn12_scales_init,  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale
     input [SRAM1024_AW-1:0] addr_cnn12_b_init,     
     input [SRAM512_AW-1:0] addr_dcnn2_w_init,    // from top.v, w0
     input [SRAM1024_AW-1:0] addr_dcnn2_scales_init, 
     input [SRAM8192_AW-1:0] addr_cnn21_w_init,    // from top.v, w0
     input [SRAM1024_AW-1:0] addr_cnn21_scales_init,  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale
     input [SRAM1024_AW-1:0] addr_cnn21_b_init,
     input [SRAM8192_AW-1:0] addr_cnn22_w_init,    // from top.v, w0
     input [SRAM1024_AW-1:0] addr_cnn22_scales_init,  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale
     input [SRAM1024_AW-1:0] addr_cnn22_b_init,     
     
     input [SRAM32_DW-1 : 0] sram1_dout,   //sram1_dout: b and scales 
     input [SRAM16_DW-1 : 0] sram4_dout,
     input [SRAM16_DW-1 : 0] sram5_dout, 
     input [SRAM16_DW-1 : 0] sram6_dout,
     input decoder_rdy, //

     input signed [DATA_DW-1:0]  mult_a_out_round,
     input signed [DATA_DW-1:0] mult_b_out_round,
     input signed [DATA_DW-1:0] pe_out_sum_a_final,
     input signed [DATA_DW-1:0] pe_out_sum_b_final,

     output reg [3-1 : 0] decoder_top_state,
     output  decoder_out_vld,
    //  output reg signed [DATA_DW-1: 0] decoder_out, // dcnn1, cnn11, cnn12
    //  output reg [2*DATA_DW-1: 0] decoder_out_cat, // cnn21, cnn22
     output reg [2-1:0] softmax_out,
     output decoder_done,    
     output [SRAM8192_AW-1:0] addr_sram, 
     output sram1_en,
     output sram4_en,
     output sram5_en,   
     output sram6_en,  
     output  signed [W_DW-1 : 0] decoder_w, //segment.v
     output  signed [SCALE_DW -1 : 0] decoder_scale,
     output  signed [SCALE_DW -1 : 0] decoder_b1,
     output  signed [SCALE_DW -1 : 0] decoder_b2, // for top_decoder_state_16
    //  output reg signed [SCALE_DW -1 : 0] decoder_b3, // for top_decoder_state_8
    //  output reg signed [SCALE_DW -1 : 0] decoder_b4, // for top_decoder_state_8
     output spad1_w_we_en, //
     output [PE_NUM-2:0] spad_w_we_en_2_32,
     output  [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_re,
     output  [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_we,
     output  [$clog2(SPAD_DEPTH)-1 : 0] spad_a_addr_re,
    output [$clog2(SPAD_DEPTH)-1 : 0] spad_a_addr_we, //new
    output  [PE_NUM-1 : 0] spad_a_we_en_1_32, //new
    output  [INPUT_DW -1 : 0] spad1_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad2_a_data_sram_in, 
    output  [DATA_DW -1 : 0] spad3_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad4_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad5_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad6_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad7_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad8_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad9_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad10_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad11_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad12_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad13_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad14_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad15_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad16_a_data_sram_in,  
    output  [DATA_DW -1 : 0] spad17_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad18_a_data_sram_in, 
    output  [DATA_DW -1 : 0] spad19_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad20_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad21_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad22_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad23_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad24_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad25_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad26_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad27_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad28_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad29_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad30_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad31_a_data_sram_in,
    output  [DATA_DW -1 : 0] spad32_a_data_sram_in,  
    output  [1:0]  mult_a_crl, // 00:idle, 01:requantize     
    output  [1:0]  mult_b_crl,
    output  [1:0] add_a_crl,
    output  [1:0] add_b_crl, 

    output [3*PE_NUM-1:0] mult_int8_crl_all,
    output mult_out_round_en,
    // output sum_a_final_en,
    // output sum_b_final_en,
    output reg cnn22_is_first_2d,
    output cnn22_is_first,
    output [$clog2(PE_NUM+1)-1:0] cnt_cho_32,
    // special for dcnn 
    output [2*PE_NUM-1:0] shift_crl_all, 
    output [PE_NUM-1:0] cnt_bt_all,
    output  dcnn1_temp_value_vld,
    output dcnn1_transfer_temp_value_en,
    output dcnn1_temp_rst,
    output signed [DATA_BQ_DW-1 : 0]  dcnn1_temp_value_for_1,
    input  [SRAM8_DW-1:0]  sram_act_dout,
    output [SRAM8192_AW -1 : 0] addr_sram_act,
    output sram_act_en,
    output sram_act_we,
    output [SRAM8_DW-1:0] sram_act_din
    );

    // localparam DCNN1_CHIN = 64;
    // localparam DCNN1_LENGTH_IN = 64;
    // localparam DCNN1_CHOUT = 32;
    // localparam DCNN_STRIDE = 2;
    // localparam DCNN_PADDING = 3;
    // localparam DCNN_KS = 8;
    // localparam DCNN1_LENGTH_OUT = DCNN_STRIDE * (DCNN1_LENGTH_IN - 1) -  2 * DCNN_PADDING + DCNN_KS;

    localparam N_DECODER_TOP = 3;
    localparam idle = 3'b000;
    // localparam reorder = 3'b001;
    localparam dcnn1 =  3'b001; //from sram
    localparam cnn11 = 3'b011;
    localparam cnn12 = 3'b111;
    localparam dcnn2 = 3'b101;
    localparam cnn21  = 3'b110 ;
    localparam cnn22 = 3'b010;
    localparam done = 3'b100; //
     
    reg [N_DECODER_TOP-1 : 0] decoder_top_state_next;
    wire layer_done;
    assign decoder_done = (decoder_top_state == done);

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            decoder_top_state <= idle;
        else
            decoder_top_state <= decoder_top_state_next;
    end

    always @(*) begin
        case(decoder_top_state)
        idle: begin
            if (decoder_rdy) decoder_top_state_next = dcnn1; // need to be changed
            else decoder_top_state_next = idle;
        end
        dcnn1: begin
            if (layer_done) decoder_top_state_next = cnn11;
            else decoder_top_state_next = dcnn1;            
        end
        cnn11:begin
            if (layer_done) decoder_top_state_next = cnn12;
            else decoder_top_state_next = cnn11;              
        end
        cnn12: begin
            if (layer_done) decoder_top_state_next = dcnn2;
            else decoder_top_state_next = cnn12;              
        end
        dcnn2:begin
            if (layer_done) decoder_top_state_next = cnn21;
            else decoder_top_state_next = dcnn2;            
        end
        cnn21: begin
            if (layer_done) decoder_top_state_next = cnn22;
            else decoder_top_state_next = cnn21;             
        end
        cnn22: begin
            if (layer_done) decoder_top_state_next = done;
            else decoder_top_state_next = cnn22;             
        end
        done: decoder_top_state_next = idle; 
        endcase
    end  

    reg layer_done_d;
    reg decoder_rdy_d;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            layer_done_d <= 0;
            decoder_rdy_d <= 0;
            end
        else begin
            layer_done_d <= layer_done;
            decoder_rdy_d <= decoder_rdy;
        end    
    end

    wire layer_rdy;
    assign layer_rdy = decoder_rdy_d | (layer_done_d & (decoder_top_state != done)); // layer_done_d is because need to back to idle

    reg [SRAM8192_AW-1:0] addr_decoder_w_init;    // from top.v, w0
    reg [SRAM1024_AW-1:0] addr_decoder_b_init;    // from top.v, b0
    reg [SRAM1024_AW-1:0] addr_decoder_scales_init;  // from top.v , SwSx_Sg, 

    always @(*) begin
        case(decoder_top_state_next)
        idle: begin
            addr_decoder_w_init = addr_dcnn1_w_init; //need to change
            addr_decoder_b_init = 0;
            addr_decoder_scales_init  =   addr_dcnn1_scales_init; 
            // addr_decoder_w_init = addr_cnn11_w_init;
            // addr_decoder_b_init = addr_cnn11_b_init;
            // addr_decoder_scales_init  =   addr_cnn11_scales_init;               
        end
        dcnn1: begin
            addr_decoder_w_init = addr_dcnn1_w_init;
            addr_decoder_b_init = 0;
            addr_decoder_scales_init  =   addr_dcnn1_scales_init;           
        end
        cnn11:begin
            addr_decoder_w_init = addr_cnn11_w_init;
            addr_decoder_b_init = addr_cnn11_b_init;
            addr_decoder_scales_init  =   addr_cnn11_scales_init;           
        end
        cnn12: begin
            addr_decoder_w_init = addr_cnn12_w_init;
            addr_decoder_b_init = addr_cnn12_b_init;
            addr_decoder_scales_init  =   addr_cnn12_scales_init;                
        end
        dcnn2: begin
            addr_decoder_w_init = {4'B0,addr_dcnn2_w_init}; //SRAM6
            addr_decoder_b_init = 0;
            addr_decoder_scales_init  =   addr_dcnn2_scales_init;            
        end
        cnn21: begin
            addr_decoder_w_init = addr_cnn21_w_init;
            addr_decoder_b_init = addr_cnn21_b_init;
            addr_decoder_scales_init  =   addr_cnn21_scales_init;            
        end
        cnn22: begin
            addr_decoder_w_init = addr_cnn22_w_init;
            addr_decoder_b_init = addr_cnn22_b_init;
            addr_decoder_scales_init  =   addr_cnn22_scales_init;            
        end
        default:begin
            addr_decoder_w_init = addr_dcnn1_w_init; //need to change
            addr_decoder_b_init = 0;
            addr_decoder_scales_init  =   addr_dcnn1_scales_init;             
        end
        endcase
    end

    //  input signed [DATA_DW-1:0]  mult_a_out_round,
    //  input signed [DATA_DW-1:0] mult_b_out_round,
    //  input signed [DATA_DW-1:0] pe_out_sum_a_final,
    //  input signed [DATA_DW-1:0] pe_out_sum_b_final,
wire signed [DATA_DW-1:0] mult_a_out_round_relu;
wire signed [DATA_DW-1:0] mult_b_out_round_relu;
reg signed [DATA_DW-1:0] mult_a_out_round_d;
reg signed [DATA_DW-1:0] mult_b_out_round_d;
assign mult_a_out_round_relu = (mult_a_out_round>0)?mult_a_out_round:0;
assign mult_b_out_round_relu = (mult_b_out_round>0)?mult_b_out_round:0;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin 
        mult_a_out_round_d <= 0;
        mult_b_out_round_d <= 0;
    end
    else begin
        if (decoder_top_state == cnn22) begin
            mult_a_out_round_d <= mult_a_out_round;
            mult_b_out_round_d <= mult_b_out_round;            
        end

    end
end
reg [1:0]softmax_out_12;
reg [1:0]softmax_out_34;
reg signed [DATA_DW-1:0] mult_a_out_round_max;
reg signed [DATA_DW-1:0] mult_b_out_round_max;
reg cnn22_is_first_d;
// reg cnn22_is_first_2d;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        cnn22_is_first_d           <= 0;
        cnn22_is_first_2d <= 0;
    end    
    else begin
        cnn22_is_first_d           <= cnn22_is_first;
        cnn22_is_first_2d <= cnn22_is_first_d;            
    end
end
always @(*) begin
    if (decoder_top_state == cnn22) begin
        if ((decoder_out_vld)& (!cnn22_is_first_2d)) begin
            if (mult_a_out_round_d > mult_a_out_round) begin
                softmax_out_12 = 2'b10;
                mult_a_out_round_max = mult_a_out_round_d;
            end
            else begin
                softmax_out_12 = 2'b01;
                mult_a_out_round_max = mult_a_out_round;
            end
            if (mult_b_out_round_d > mult_b_out_round) begin
                softmax_out_34 = 2'b10;
                mult_b_out_round_max = mult_b_out_round_d;
            end
            else begin
                softmax_out_34 = 2'b01;
                mult_b_out_round_max = mult_b_out_round;
            end        
            if (mult_a_out_round_max > mult_b_out_round_max) begin
                if (softmax_out_12 ==  2'b10) softmax_out = 4'b00;
                else if  (softmax_out_12 ==  2'b01) softmax_out = 4'b10;
            end
            else begin
                if (softmax_out_34 ==  2'b10) softmax_out = 4'b01;
                else if  (softmax_out_34 ==  2'b01) softmax_out = 4'b11;
            end       
        end
        else begin
            softmax_out_12 = 0;
            softmax_out_34 = 0;
            softmax_out = 0;
            mult_a_out_round_max = 0;
            mult_b_out_round_max = 0;
        end
    end
    else begin
        softmax_out_12 = 0;
        softmax_out_34 = 0;
        softmax_out = 0;
        mult_a_out_round_max = 0;
        mult_b_out_round_max = 0;
    end
end
reg signed [DATA_DW-1: 0] decoder_out;
reg [2*DATA_DW-1: 0] decoder_out_cat;
always @(*) begin
    if (decoder_top_state == dcnn1)  begin
        decoder_out = mult_a_out_round;
        decoder_out_cat = 0;

    end
    else if ((decoder_top_state == cnn11) |(decoder_top_state == cnn12)) begin
        decoder_out = mult_a_out_round_relu;
        decoder_out_cat = 0;        
    end
    else if ((decoder_top_state == dcnn2) ) begin
        decoder_out  = 0;
        decoder_out_cat =  {mult_b_out_round,mult_a_out_round};
    end
    else if ((decoder_top_state == cnn21)) begin
        decoder_out  = 0;
        decoder_out_cat = {mult_b_out_round_relu,mult_a_out_round_relu};        
    end
    else begin
        decoder_out = 0;
        decoder_out_cat =0;
    end

    
end
DECODER_L#(
.DATA_DW(DATA_DW),
.INPUT_DW(INPUT_DW),
.DATA_BQ_DW(DATA_BQ_DW),
.W_DW(W_DW),
.B_DW(B_DW),
.SCALE_DW(SCALE_DW),
.SRAM1024_AW(SRAM1024_AW),
.SRAM8192_AW(SRAM8192_AW),
.SRAM512_AW(SRAM512_AW),
.SRAM32_DW(SRAM32_DW),
.SRAM16_DW(SRAM16_DW),
.SRAM8_DW(SRAM8_DW),
.SPAD_DEPTH(SPAD_DEPTH),
.PE_NUM(PE_NUM),
.DCNN1_CHIN(DCNN1_CHIN),
.DCNN1_LENGTH_IN(DCNN1_LENGTH_IN),
.DCNN1_CHOUT(DCNN1_CHOUT),
.DCNN_PADDING(DCNN_PADDING),
.DCNN_KS(DCNN_KS),
.DCNN_STRIDE(DCNN_STRIDE),
.DCNN1_LENGTH_OUT(DCNN1_LENGTH_OUT),
.CNN11_LENGTH_IN(CNN11_LENGTH_IN),
.CNN11_CHIN(CNN11_CHIN),
.CNN11_CHOUT(CNN11_CHOUT),
.CNN_PADDING(CNN_PADDING),
.CNN_KS(CNN_KS),
.CNN11_LENGTH_OUT(CNN11_LENGTH_OUT),
.CNN12_LENGTH_IN(CNN12_LENGTH_IN),
.CNN12_CHIN(CNN12_CHIN),
.CNN12_CHOUT(CNN12_CHOUT),
.CNN12_LENGTH_OUT(CNN12_LENGTH_OUT),
.DCNN2_LENGTH_IN(DCNN2_LENGTH_IN),
.DCNN2_CHIN(DCNN2_CHIN),
.DCNN2_CHOUT(DCNN2_CHOUT),
.DCNN2_LENGTH_OUT(DCNN2_LENGTH_OUT),
.CNN21_LENGTH_IN(CNN21_LENGTH_IN),
.CNN21_CHIN(CNN21_CHIN),
.CNN21_CHOUT(CNN21_CHOUT),
.CNN21_LENGTH_OUT(CNN21_LENGTH_OUT),
.CNN22_LENGTH_IN(CNN22_LENGTH_IN),
.CNN22_CHIN(CNN22_CHIN),
.CNN22_CHOUT(CNN22_CHOUT),
.CNN22_LENGTH_OUT(CNN22_LENGTH_OUT),
.ADDR_LSTM10_SRAM_ACT_INIT(ADDR_LSTM10_SRAM_ACT_INIT),
.ADDR_LSTM11_SRAM_ACT_INIT(ADDR_LSTM11_SRAM_ACT_INIT))
decoder_l(
    .wclk(wclk),
    .sclk(sclk),
    .rst_n(rst_n),
    // .act_sr2(act_sr2),
    .act_sr1_1(act_sr1_1),
    // .act_sr3(act_sr3),
    // .act_sr4(act_sr4),
    .addr_decoder_w_init(addr_decoder_w_init),
    .addr_decoder_scales_init(addr_decoder_scales_init),
    .addr_decoder_b_init(addr_decoder_b_init),
    .sram1_dout(sram1_dout),
    .sram4_dout(sram4_dout),
    .sram5_dout(sram5_dout),
    .sram6_dout(sram6_dout),
    .layer_rdy(layer_rdy),
    .decoder_top_state(decoder_top_state),
    .shift_crl_all(shift_crl_all),
    .cnn22_is_first(cnn22_is_first),
    .cnt_cho_32_3d(cnt_cho_32),
    .layer_out_vld(decoder_out_vld),
    .addr_sram(addr_sram),
    .sram1_en(sram1_en),
    .sram4_en(sram4_en),
    .sram5_en(sram5_en),
    .sram6_en(sram6_en),
    .decoder_w(decoder_w),
    .decoder_scale(decoder_scale),
    .decoder_b1(decoder_b1),
    .decoder_b2(decoder_b2),
    // .decoder_b3(decoder_b3),
    // .decoder_b4(decoder_b4),
    .layer_done(layer_done),
    .spad1_w_we_en(spad1_w_we_en),
    .spad_w_we_en_2_32(spad_w_we_en_2_32),
    .spad_w_addr_re(spad_w_addr_re),
    .spad_w_addr_we(spad_w_addr_we),
    .spad_a_addr_re(spad_a_addr_re),
    .spad_a_addr_we(spad_a_addr_we),
    .spad_a_we_en_1_32(spad_a_we_en_1_32),
    .spad1_a_data_sram_in(spad1_a_data_sram_in),
    .spad2_a_data_sram_in(spad2_a_data_sram_in),
    .spad3_a_data_sram_in(spad3_a_data_sram_in),
    .spad4_a_data_sram_in(spad4_a_data_sram_in),
    .spad5_a_data_sram_in(spad5_a_data_sram_in),
    .spad6_a_data_sram_in(spad6_a_data_sram_in),
    .spad7_a_data_sram_in(spad7_a_data_sram_in),
    .spad8_a_data_sram_in(spad8_a_data_sram_in),
    .spad9_a_data_sram_in(spad9_a_data_sram_in),
    .spad10_a_data_sram_in(spad10_a_data_sram_in),
    .spad11_a_data_sram_in(spad11_a_data_sram_in),
    .spad12_a_data_sram_in(spad12_a_data_sram_in),
    .spad13_a_data_sram_in(spad13_a_data_sram_in),
    .spad14_a_data_sram_in(spad14_a_data_sram_in),
    .spad15_a_data_sram_in(spad15_a_data_sram_in),
    .spad16_a_data_sram_in(spad16_a_data_sram_in),
    .spad17_a_data_sram_in(spad17_a_data_sram_in),
    .spad18_a_data_sram_in(spad18_a_data_sram_in),
    .spad19_a_data_sram_in(spad19_a_data_sram_in),
    .spad20_a_data_sram_in(spad20_a_data_sram_in),   
    .spad21_a_data_sram_in(spad21_a_data_sram_in),
    .spad22_a_data_sram_in(spad22_a_data_sram_in),
    .spad23_a_data_sram_in(spad23_a_data_sram_in),
    .spad24_a_data_sram_in(spad24_a_data_sram_in),
    .spad25_a_data_sram_in(spad25_a_data_sram_in),
    .spad26_a_data_sram_in(spad26_a_data_sram_in),
    .spad27_a_data_sram_in(spad27_a_data_sram_in),
    .spad28_a_data_sram_in(spad28_a_data_sram_in),
    .spad29_a_data_sram_in(spad29_a_data_sram_in),
    .spad30_a_data_sram_in(spad30_a_data_sram_in),
    .spad31_a_data_sram_in(spad31_a_data_sram_in),
    .spad32_a_data_sram_in(spad32_a_data_sram_in),

    .mult_a_crl(mult_a_crl),
    .mult_b_crl(mult_b_crl),
    .add_a_crl(add_a_crl),
    .add_b_crl(add_b_crl),
    .mult_int8_crl_all(mult_int8_crl_all),
    .mult_out_round_en(mult_out_round_en),
    // .sum_a_final_en(sum_a_final_en),
    // .sum_b_final_en(sum_b_final_en),
    .dcnn1_temp_value_vld(dcnn1_temp_value_vld),
    .dcnn1_transfer_temp_value_en(dcnn1_transfer_temp_value_en),
    .dcnn1_temp_rst(dcnn1_temp_rst),
    .cnt_bt_all(cnt_bt_all),
    .dcnn1_temp_value_for_1(dcnn1_temp_value_for_1),
    .sram_act_dout(sram_act_dout),
    .addr_sram_act(addr_sram_act),
    .sram_act_en(sram_act_en),
    .sram_act_we(sram_act_we),
    .sram_act_din(sram_act_din),
    .decoder_out(decoder_out),
    .decoder_out_cat(decoder_out_cat)
);

endmodule