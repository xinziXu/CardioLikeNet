`timescale  1ns/100ps
module SEG_NETWORK #(parameter ENCODER_WB_DW = 32,
                     SRAM16_DW = 16,
                     SRAM8_DW = 8,
                     SRAM32_DW = 32,
                     SRAM1024_AW = 10,
                     SRAM8192_AW = 13,
                     SRAM512_AW = 9,
                     INPUT_DW = 12,
                     DATA_OUT_DW = 8,
                     DATA_BQ_DW = 32,
                     LSTM_WU_DW = 8,
                     LSTM_B_DW = 32,
                     DECODER_W_DW = 8,
                     DECODER_SCALE_DW = 32,
                     DECODER_B_DW = 32,
                     SPAD_DEPTH = 8,
                     ENCODER_SCALE_DW = 16,
                     LSTM_SCALE_DW = 24,
                     PE_NUM  = 32,
                     ENCODER_LENGTH_IN = 256,
                     ENCODER_STRIDE = 4,
                     DCNN1_LENGTH_OUT =  ENCODER_LENGTH_IN/2,
                     CNN22_LENGTH_OUT =  ENCODER_LENGTH_IN
                     )
                   (input sclk,
                     input wclk,
                     input rst_n,
                     input network_rdy,
                     input [INPUT_DW*ENCODER_LENGTH_IN-1:0] input_signal,                  
                     output [3:0] seg_state,

                     input [SRAM32_DW-1: 0] sram1_dout,
                     input [SRAM16_DW-1: 0] sram2_dout,
                     input [SRAM16_DW-1: 0] sram3_dout,
                     input [SRAM16_DW-1: 0] sram4_dout,
                     input [SRAM16_DW-1: 0] sram5_dout,
                     input [SRAM16_DW-1: 0] sram6_dout,
                     output reg [SRAM8192_AW-1: 0] addr_sram,
                     output reg sram1_en,    
                     output reg sram2_en,  
                     output reg sram3_en, 
                     output reg sram4_en,  
                     output reg sram5_en,  
                     output reg sram6_en, 
                     output [2:0] lstm_top_state, 
                     output [2:0] decoder_top_state,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  // 1 is enabled, from top                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  // 1 is read, 0 is write, from top
                     input [SRAM1024_AW-1:0] addr_encoder_w_init,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         // from top.v
                     input [SRAM1024_AW-1:0] addr_encoder_b_init,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         // from top.v
                     input [SRAM1024_AW-1:0] addr_encoder_output_scale,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   // from top.v  
                     input [SRAM8192_AW-1:0] addr_lstm_w00_init,    // from top.v, w0
                     input [SRAM8192_AW-1:0] addr_lstm_u00_init,    // from top.v, w0
                     input [SRAM1024_AW-1:0] addr_lstm_b00_init,    // from top.v, b0
                     input [SRAM1024_AW-1:0] addr_lstm_scales_00_init,  // from top.v , SwSx_Sg, SuSh_Sg_00, SiSg_Sc_00, SoSc_Sh,ct_scale,gates_scale
                     input [SRAM8192_AW-1:0] addr_lstm_w01_init,    // from top.v, w0
                     input [SRAM8192_AW-1:0] addr_lstm_u01_init,    // from top.v, w0
                     input [SRAM1024_AW-1:0] addr_lstm_b01_init,    // from top.v, b0
                     input [SRAM1024_AW-1:0] addr_lstm_scales_01_init,                      
                     input [SRAM8192_AW-1:0] addr_lstm_w10_init,    // from top.v, w0
                     input [SRAM8192_AW-1:0] addr_lstm_u10_init,    // from top.v, w0
                     input [SRAM1024_AW-1:0] addr_lstm_b10_init,    // from top.v, b0
                     input [SRAM1024_AW-1:0] addr_lstm_scales_10_init,  
                     input [SRAM8192_AW-1:0] addr_lstm_w11_init,    // from top.v, w0
                     input [SRAM8192_AW-1:0] addr_lstm_u11_init,    // from top.v, w0
                     input [SRAM1024_AW-1:0] addr_lstm_b11_init,    // from top.v, b0
                     input [SRAM1024_AW-1:0] addr_lstm_scales_11_init, 
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

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               // from top.v
                     output reg spad1_w_we_en,// the enable signal for the first spad
                     output reg [PE_NUM-2:0] spad_w_we_en_2_32, //the enable signal for the rest of PEs
                     
                     output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_re, 
                     output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_we, 
                     output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_a_addr_re, 
                    //  output reg [$clog2(SPAD_DEPTH)*PE_NUM-1 : 0] spad_a_addr_we_all,\
                     output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_a_addr_we,
                     output reg [PE_NUM-1 : 0] spad_a_we_en_all,
                     output reg [INPUT_DW -1 : 0] spad1_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad2_a_data_sram_in, 
                     output reg [DATA_OUT_DW -1 : 0] spad3_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad4_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad5_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad6_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad7_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad8_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad9_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad10_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad11_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad12_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad13_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad14_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad15_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad16_a_data_sram_in, 
                     output reg [DATA_OUT_DW -1 : 0] spad17_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad18_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad19_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad20_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad21_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad22_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad23_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad24_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad25_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad26_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad27_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad28_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad29_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad30_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad31_a_data_sram_in,
                     output reg [DATA_OUT_DW -1 : 0] spad32_a_data_sram_in,

                     output reg [INPUT_DW*SPAD_DEPTH -1 : 0] spad1_a_data_in, 
                     output reg [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad17_a_data_in,
                     output reg [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad18_a_data_in,
                     output reg [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad19_a_data_in,
                     output reg [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad20_a_data_in,
                     output reg [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad21_a_data_in,
                     output reg [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad22_a_data_in,
                     output reg [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad23_a_data_in,
                     output reg [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad24_a_data_in,
                     output reg [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad25_a_data_in,
                     output reg [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad26_a_data_in,
                     output reg [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad27_a_data_in,
                     output reg [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad28_a_data_in,
                     output reg [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad29_a_data_in,
                     output reg [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad30_a_data_in,  
                     output reg [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad31_a_data_in,
                     output reg [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad32_a_data_in,
                     output signed [ENCODER_WB_DW-1 : 0] encoder_b, 
                     output signed [ENCODER_WB_DW-1 : 0] encoder_w, 
                     output signed [LSTM_WU_DW-1 : 0] lstm_wu,
                     input signed [DATA_OUT_DW-1: 0] encoder_out, //encoder ,change
                     output signed [DATA_BQ_DW-1 : 0]  dcnn1_temp_value_for_1,
                     output signed [DECODER_W_DW-1:0] decoder_w, // seg-network out, spad-w input
                     output signed [DECODER_B_DW-1:0] decoder_b1, //seg-network out, pe-main input
                     output signed [DECODER_B_DW-1:0] decoder_b2, //seg-network out, pe-main input


                     output cnn22_is_first,
                     output cnn22_is_first_2d,
                     input [PE_NUM*DATA_BQ_DW-1: 0] pe_out_32b_all,
                     input signed [2*DATA_OUT_DW+LSTM_SCALE_DW-1: 0] pe_out_a,
                     input signed [2*DATA_OUT_DW+LSTM_SCALE_DW-1: 0] pe_out_b,
                     input signed [DATA_OUT_DW-1: 0] mult_a_out_round, // pe-main out, seg-network in
                     input signed [DATA_OUT_DW-1: 0] mult_b_out_round,// pe-main out, seg-network in
                     
                     input signed [DATA_OUT_DW-1:0] pe_out_sum_a_final,
                     input signed [DATA_OUT_DW-1:0] pe_out_sum_b_final,
                    //  input signed [DATA_OUT_DW-1:0]  out_temp_A_final,
                     output signed [2*(2*DATA_OUT_DW+LSTM_SCALE_DW)-1: 0] lstm_ct_temp_out_cat,
                     
                     output signed [PE_NUM * DATA_OUT_DW-1: 0] hardmard_a_all,
                     output signed [PE_NUM * DATA_OUT_DW-1: 0] hardmard_b_all,
                     output signed [LSTM_B_DW-1: 0] lstm_b,
                     output reg signed [DATA_BQ_DW-1:0] out_bq, // lstm: from each PEs
                     output reg signed [LSTM_SCALE_DW -1 : 0] scale,
                     output reg signed [DATA_BQ_DW-1:0] out_bq2, // lstm: from each PEs
                     output reg signed [LSTM_SCALE_DW -1 : 0] scale2,
                    //  output signed [2*DATA_OUT_DW-1: 0]  lstm_ct_temp_cat,
                     output signed [ENCODER_SCALE_DW -1 : 0] encoder_scale, //encoder
                    //  output reg signed [2*DATA_OUT_DW-1:0] seg_out, //temp

                     output  reg [1:0] mult_a_crl, 
                     output reg [1:0] mult_b_crl,
                     output  reg [1:0] add_a_crl,
                     output reg [1:0] add_b_crl,                      // 110: hardmard_p 
                     output reg [2:0] mult_int8_crl_1_16, 
                     output reg [2:0] mult_int8_crl_17_32,
                     output [3*PE_NUM-1:0] decoder_mult_int8_crl, // all the pes
                     output reg mult_out_round_en,
                     output reg pe_out_sum_a_final_en,
                     output reg pe_out_sum_b_final_en,
                     output encoder_relu_en,
                     output encoder_round_en,
                     output decoder_done,

                     output [1:0] softmax_out,
                     output decoder_out_vld,
                     output dcnn1_temp_value_vld,
                     output dcnn1_transfer_temp_value_en,
                     output dcnn1_temp_rst,
                     output [1:0] encoder_shift_en,
                    //  output lstm_xt_shift_en,
                    //  output [2*PE_NUM-1:0] shift_crl_all,
                    //  output [PE_NUM-1:0] cnt_bt_all,
                     input [DATA_BQ_DW-1:0] act_sr1_1,
                    //  input [ACTIVATION_BUF_LEN2-1:0] act_sr2,
                    //  input [DATA_OUT_DW*ACTIVATION_BUF_LEN3-1:0] act_sr3,
                    //  input [ACTIVATION_BUF_LEN4-1:0] act_sr4
                    //  input [SRAM8192_AW - 1:0] addr_sram_act_test,
                    //  input [SRAM8_DW-1:0] sram_act_din_test,
                    //  input sram_act_we_test
                    input [SRAM8_DW-1:0] sram_act_dout,
                    output reg [SRAM8_DW-1:0] sram_act_din,
                    output reg sram_act_en,
                    output reg sram_act_we,
                    output reg [SRAM8192_AW - 1:0] addr_sram_act
                    );



    localparam dcnn1 =  3'b001; //from sram
    localparam cnn11 = 3'b011;
    localparam cnn12 = 3'b111;
    localparam dcnn2 = 3'b101;
    localparam cnn21  = 3'b110 ;
    localparam cnn22 = 3'b010;
    wire encoder_done;
    wire lstm_done;



    localparam ENCODER_PADDING_PRE = 3;
    localparam ENCODER_PADDING_POST = 1;
    localparam ENCODER_KS = 8;
    localparam ENCODER_CHIN = 1;
    localparam ENCODER_CHOUT = 32;
    localparam ENCODER_LENGTH_OUT = 64;
    localparam LSTM_NUM_LAYERS = 2;
    localparam LSTM_NUM_DIR = 2;
    localparam LSTM_HS = 32;
    localparam DCNN1_CHOUT = 32;
    localparam DCNN_STRIDE = 2;
    localparam DCNN_PADDING = 3;
    localparam DCNN_KS = 8;
    localparam CNN11_LENGTH_IN = DCNN1_LENGTH_OUT;
    localparam CNN11_CHIN = DCNN1_CHOUT;
    localparam CNN11_CHOUT = 32;
    localparam CNN_PADDING = 2;
    localparam CNN_KS = 5;
    localparam CNN11_LENGTH_OUT = DCNN1_LENGTH_OUT;
    localparam CNN12_LENGTH_IN = CNN11_LENGTH_OUT;
    localparam CNN12_CHIN = CNN11_CHOUT;
    localparam CNN12_CHOUT = 16;
    localparam CNN12_LENGTH_OUT =  DCNN1_LENGTH_OUT;
    localparam DCNN2_LENGTH_IN = CNN12_LENGTH_OUT;
    localparam DCNN2_CHIN = CNN12_CHOUT;
    localparam DCNN2_CHOUT =  8;
    localparam DCNN2_LENGTH_OUT = DCNN_STRIDE * (DCNN2_LENGTH_IN - 1) -  2 * DCNN_PADDING + DCNN_KS;
    localparam CNN21_LENGTH_IN = DCNN2_LENGTH_OUT;
    localparam CNN21_CHIN = DCNN2_CHOUT;
    localparam CNN21_CHOUT = 8;
    localparam CNN21_LENGTH_OUT = DCNN2_LENGTH_OUT;
    localparam CNN22_LENGTH_IN = DCNN2_LENGTH_OUT;
    localparam CNN22_CHIN = CNN21_CHOUT;
    localparam CNN22_CHOUT = 4;




    wire encoder_spad1_w_we_en;// the enable signal for the first spad
    wire lstm_spad1_w_we_en;// the enable signal for the first spad
    wire [PE_NUM-2:0] lstm_spad_w_we_en_2_32; //the enable signal for the rest of PEs

    wire decoder_spad1_w_we_en;// the enable signal for the first spad
    wire [PE_NUM-2:0] decoder_spad_w_we_en_2_32; //the enable signal for the rest of PEs

    wire [$clog2(SPAD_DEPTH)-1 : 0] lstm_spad_w_addr_re; 
    wire [$clog2(SPAD_DEPTH)-1 : 0] encoder_spad_w_addr_re; 
    wire [$clog2(SPAD_DEPTH)-1 : 0] decoder_spad_w_addr_re; 
    wire [$clog2(SPAD_DEPTH)-1 : 0] lstm_spad_w_addr_we;
    wire [$clog2(SPAD_DEPTH)-1 : 0] decoder_spad_w_addr_we;
    wire [$clog2(SPAD_DEPTH)-1 : 0] encoder_spad_w_addr_we;  
    wire [$clog2(SPAD_DEPTH)-1 : 0] lstm_spad_a_addr_re; 
    wire [$clog2(SPAD_DEPTH)-1 : 0] decoder_spad_a_addr_re; 
    wire [$clog2(SPAD_DEPTH)-1 : 0] encoder_spad_a_addr_re; 
    wire [INPUT_DW*SPAD_DEPTH -1 : 0] encoder_spad1_a_data_in;
    // wire [INPUT_DW*SPAD_DEPTH -1 : 0] lstm_spad1_a_data_in;
    // wire [INPUT_DW*SPAD_DEPTH -1 : 0] decoder_spad1_a_data_in;
    // wire [$clog2(SPAD_DEPTH)*PE_NUM/2-1 : 0] lstm_spad_a_addr_we_1_16; //new
    wire [$clog2(SPAD_DEPTH)-1 : 0] lstm_spad_a_addr_we;
    wire [PE_NUM/2-1 : 0] lstm_spad_a_we_en_1_16; //new
    wire [INPUT_DW -1 : 0] lstm_spad1_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] lstm_spad2_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] lstm_spad3_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] lstm_spad4_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] lstm_spad5_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] lstm_spad6_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] lstm_spad7_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] lstm_spad8_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] lstm_spad9_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] lstm_spad10_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] lstm_spad11_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] lstm_spad12_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] lstm_spad13_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] lstm_spad14_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] lstm_spad15_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] lstm_spad16_a_data_sram_in;


    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] lstm_spad17_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] lstm_spad18_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] lstm_spad19_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] lstm_spad20_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] lstm_spad21_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] lstm_spad22_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] lstm_spad23_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] lstm_spad24_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] lstm_spad25_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] lstm_spad26_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] lstm_spad27_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] lstm_spad28_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] lstm_spad29_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] lstm_spad30_a_data_in;  
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] lstm_spad31_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] lstm_spad32_a_data_in;

    wire [$clog2(SPAD_DEPTH)-1 : 0] decoder_spad_a_addr_we;
    wire [PE_NUM-1 : 0] decoder_spad_a_we_en_1_32; //new
    wire [INPUT_DW -1 : 0] decoder_spad1_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad2_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad3_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad4_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad5_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad6_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad7_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad8_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad9_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad10_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad11_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad12_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad13_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad14_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad15_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad16_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad17_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad18_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad19_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad20_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad21_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad22_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad23_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad24_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad25_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad26_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad27_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad28_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad29_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad30_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad31_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] decoder_spad32_a_data_sram_in;



    wire signed [DATA_BQ_DW-1:0] lstm_out_bq; // lstm: from each PEs
    wire signed [LSTM_SCALE_DW -1 : 0] lstm_scale;
    wire signed [DATA_BQ_DW-1:0] lstm_out_bq2; // lstm: from each PEs
    wire signed [LSTM_SCALE_DW -1 : 0] lstm_scale2;
    wire signed [DECODER_SCALE_DW-1:0] decoder_scale;


    wire lstm_mult_out_round_en;
    wire decoder_mult_out_round_en;
    wire lstm_sum_a_final_en;
    // wire decoder_sum_a_final_en;
    wire lstm_sum_b_final_en;
    // wire decoder_sum_b_final_en;    
    wire [1:0] lstm_mult_a_crl; 
    wire [1:0] encoder_mult_a_crl; 
    wire [1:0] decoder_mult_a_crl; 
    wire [1:0] lstm_mult_b_crl;
    wire [1:0] decoder_mult_b_crl; 
    wire [1:0] lstm_add_a_crl;
    wire [1:0] lstm_add_b_crl; 
    wire [1:0] decoder_add_a_crl;
    wire [1:0] decoder_add_b_crl;                     // 110: hardmard_p 
    wire [2:0] lstm_mult_int8_crl_1_16; 
    wire [2:0] lstm_mult_int8_crl_17_32; 
    
 
    wire lstm_sram2_en;
    wire lstm_sram1_en;
    wire lstm_sram3_en;
    wire lstm_sram4_en;    
    wire encoder_sram1_en;
    wire decoder_sram1_en;
    wire decoder_sram4_en;
    wire decoder_sram5_en;
    wire decoder_sram6_en;


    // reg [DATA_OUT_DW*ACTIVATION_BUF_LEN1-1:0] act_sr1;
    // reg [DATA_OUT_DW*ACTIVATION_BUF_LEN2-1:0] act_sr2;
    // reg [DATA_OUT_DW*ACTIVATION_BUF_LEN3-1:0] act_sr3;
    // reg [DATA_OUT_DW*ACTIVATION_BUF_LEN4-1:0] act_sr4;
    // Network FSM
    
    localparam N       = 4;
    localparam idle    = 4'b0000;
    localparam encoder = 4'b0001;
    localparam lstm    = 4'b0010;
    localparam decoder = 4'b0100;
    localparam done    = 4'b1000;
    
    reg         [N-1:0]        seg_state_c         ; // current state
    reg         [N-1:0]        seg_state_n         ; // next state
    assign seg_state = seg_state_c;
    // encoder
    wire [SRAM1024_AW-1: 0] addr_encoder_wb;    // sram_addr
    wire [SRAM8192_AW-1: 0] addr_sram_lstm;    // sram_addr
    wire [SRAM8192_AW-1: 0] addr_sram_decoder;    // sram_addr
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            seg_state_c <= idle;
        else
            seg_state_c <= seg_state_n;
    end
    
    
    always @(*) begin
        case (seg_state_c)
            idle: begin
                if (network_rdy)
                    seg_state_n = encoder; //need to change
                else
                    seg_state_n = idle;
            end
            encoder: begin
                if (encoder_done)
                    seg_state_n = lstm;
                else
                    seg_state_n = encoder;
            end
            lstm: begin
                if (lstm_done)
                    seg_state_n = decoder;
                else
                    seg_state_n = lstm;                
            end
            decoder: begin
                if (decoder_done)
                    seg_state_n = done;
                else
                    seg_state_n = decoder;                
            end
            done:
            seg_state_n         = idle;
            default:seg_state_n = idle;
        endcase
    end
    /////////////////////////// SRAM_ACT ////////////////////////////
    // localparam  SRAM8_DW = 8;
    localparam ADDR_ENCODER_SRAM_ACT_INIT = 0;
    localparam ADDR_LSTM10_SRAM_ACT_INIT  = ENCODER_LENGTH_OUT * ENCODER_CHOUT + 2 * ENCODER_LENGTH_OUT * LSTM_HS;
    localparam ADDR_LSTM11_SRAM_ACT_INIT = 0;
    // wire [SRAM8_DW-1:0] sram_act_dout;
    // reg [SRAM8_DW-1:0] sram_act_din;
    // reg sram_act_en;
    // reg sram_act_we;
    // reg [SRAM8192_AW - 1:0] addr_sram_act;





    wire [SRAM8192_AW - 1:0] addr_encoder_sram_act;
    wire [SRAM8_DW-1:0] encoder_sram_act_din;
    wire encoder_sram_act_en;
    wire encoder_sram_act_we;

    wire [SRAM8192_AW - 1:0] addr_lstm_sram_act;
    wire [SRAM8_DW-1:0] lstm_sram_act_din;
    wire lstm_sram_act_en;
    wire lstm_sram_act_we;   

    wire [SRAM8192_AW - 1:0] addr_decoder_sram_act;
    wire [SRAM8_DW-1:0] decoder_sram_act_din;
    wire decoder_sram_act_en;
    wire decoder_sram_act_we;     
    // // Outputs: addr_sram
    
    // SOME control signals
    always @(*) begin
        case (seg_state_c)
            idle: 
            begin
                addr_sram = 0;
                sram1_en = 0;
                sram2_en = 0;
                sram3_en = 0;
                sram4_en = 0;
                sram5_en = 0;
                sram6_en = 0;
                spad1_w_we_en = 0;
                spad_w_addr_re = 0;
                spad_w_addr_we = 0;
                spad_a_addr_re = 0;
                spad1_a_data_in = 0;
                mult_a_crl = 0;
                mult_b_crl = 0;
                // seg_out_vld = 0;      // yet to be updated
                // seg_out = 0;
                // sram_act_din = sram_act_din_test; // need to be change
                // sram_act_en = 1; // need to be change
                // sram_act_we = sram_act_we_test;
                // addr_sram_act = addr_sram_act_test;
                sram_act_din = 0; 
                sram_act_en = 0; 
                sram_act_we = 0;
                addr_sram_act = 0;                
            end
            encoder: 
            begin
                addr_sram    = addr_encoder_wb;
                sram1_en = encoder_sram1_en;
                sram2_en = 0;
                sram3_en = 0;
                sram4_en = 0;
                sram5_en = 0;
                sram6_en = 0;
                spad1_w_we_en = encoder_spad1_w_we_en;
                spad_w_addr_re = encoder_spad_w_addr_re;
                spad_w_addr_we = encoder_spad_w_addr_we;
                spad_a_addr_re = encoder_spad_a_addr_re;
                spad1_a_data_in = encoder_spad1_a_data_in; //INPUT_DW - DATA_OUT_DW
                mult_a_crl = encoder_mult_a_crl;
                mult_b_crl = encoder_mult_a_crl;
                // seg_out_vld = encoder_out_vld;
                // seg_out = {{(DATA_OUT_DW){encoder_out[DATA_OUT_DW-1]}},encoder_out};
                sram_act_din = encoder_sram_act_din;
                sram_act_en = encoder_sram_act_en;
                sram_act_we = encoder_sram_act_we;
                addr_sram_act = addr_encoder_sram_act;
            end
            lstm: 
            begin
                addr_sram    = addr_sram_lstm;
                sram1_en = lstm_sram1_en;
                sram2_en = lstm_sram2_en;
                sram3_en = lstm_sram3_en;
                sram4_en = lstm_sram4_en;
                sram5_en = 0;
                sram6_en = 0;
                spad1_w_we_en = lstm_spad1_w_we_en;
                spad_w_addr_re = lstm_spad_w_addr_re;
                spad_w_addr_we = lstm_spad_w_addr_we;
                spad_a_addr_re = lstm_spad_a_addr_re;
                spad1_a_data_in = 0; //INPUT_DW - DATA_OUT_DW
                mult_a_crl = lstm_mult_a_crl;
                mult_b_crl = lstm_mult_b_crl;
                // seg_out_vld = lstm_hidden_unit_vld;
                // seg_out = lstm_hidden_cat;
                sram_act_din = lstm_sram_act_din;
                sram_act_en = lstm_sram_act_en;
                sram_act_we = lstm_sram_act_we;
                addr_sram_act = addr_lstm_sram_act;
            end
            decoder: 
            begin
                addr_sram    = addr_sram_decoder;
                sram1_en = decoder_sram1_en;
                sram2_en = 0;
                sram3_en = 0;
                sram4_en = decoder_sram4_en;
                sram5_en = decoder_sram5_en;
                sram6_en = decoder_sram6_en;
                spad1_w_we_en = decoder_spad1_w_we_en;
                spad_w_addr_re = decoder_spad_w_addr_re;
                spad_w_addr_we = decoder_spad_w_addr_we;
                spad_a_addr_re = decoder_spad_a_addr_re;
                // spad1_a_data_in = decoder_spad1_a_data_in; //INPUT_DW - DATA_OUT_DW
                spad1_a_data_in = 0;
                mult_a_crl = decoder_mult_a_crl;
                mult_b_crl = decoder_mult_b_crl;
                // seg_out_vld = decoder_out_vld;
                // seg_out = ((decoder_top_state == dcnn1)|(decoder_top_state == cnn11)|(decoder_top_state == cnn12))?{{(DATA_OUT_DW){decoder_out[DATA_OUT_DW-1]}}, decoder_out} :
                //             decoder_out_cat; // need to modify
                sram_act_din = decoder_sram_act_din;
                sram_act_en = decoder_sram_act_en;
                sram_act_we = decoder_sram_act_we;
                addr_sram_act = addr_decoder_sram_act;
            end
            done: 
            begin
                addr_sram = 0;
                sram1_en = 0;
                sram2_en = 0;
                sram3_en = 0;
                sram4_en = 0;
                sram5_en = 0;
                sram6_en = 0;
                spad1_w_we_en = 0;
                spad_w_addr_re = 0;
                spad_w_addr_we = 0;
                spad_a_addr_re = 0;
                spad1_a_data_in = 0;
                mult_a_crl = 0;
                mult_b_crl = 0;
                // seg_out_vld = 0;
                // seg_out = 0;
                sram_act_din = 0;
                sram_act_en = 0;
                sram_act_we = 0;
                addr_sram_act = 0;
            end
            default: begin
                addr_sram = 0;
                sram1_en = 0;
                sram2_en = 0;
                sram3_en = 0;
                sram4_en = 0;
                sram5_en = 0;
                sram6_en = 0;
                spad1_w_we_en = 0;
                spad_w_addr_re = 0;
                spad_w_addr_we = 0;
                spad_a_addr_re = 0;
                spad1_a_data_in = 0;
                mult_a_crl = 0;
                mult_b_crl = 0;
                // seg_out_vld = 0;
                // seg_out   = 0;
                sram_act_din = 0;
                sram_act_en = 0;
                sram_act_we = 0;
                addr_sram_act = 0;
            end
        endcase
    end
    
    always @(*) begin
        if (seg_state_c == lstm) begin
            spad_w_we_en_2_32 = lstm_spad_w_we_en_2_32;
            
            add_a_crl = lstm_add_a_crl;
            add_b_crl = lstm_add_b_crl;                      // 110: hardmard_p 
            mult_int8_crl_1_16  = lstm_mult_int8_crl_1_16;
            mult_int8_crl_17_32  = lstm_mult_int8_crl_17_32;
            out_bq = lstm_out_bq; // lstm: from each PEs
            scale = lstm_scale;
            out_bq2 = lstm_out_bq2; // lstm: from each PEs
            scale2 = lstm_scale2;    
            spad17_a_data_in =  lstm_spad17_a_data_in;
            spad18_a_data_in = lstm_spad18_a_data_in;
            spad19_a_data_in = lstm_spad19_a_data_in;
            spad20_a_data_in = lstm_spad20_a_data_in;
            spad21_a_data_in = lstm_spad21_a_data_in;
            spad22_a_data_in  = lstm_spad22_a_data_in;
            spad23_a_data_in = lstm_spad23_a_data_in;
            spad24_a_data_in = lstm_spad24_a_data_in;
            spad25_a_data_in = lstm_spad25_a_data_in;
            spad26_a_data_in  = lstm_spad26_a_data_in;
            spad27_a_data_in = lstm_spad27_a_data_in;
            spad28_a_data_in = lstm_spad28_a_data_in;
            spad29_a_data_in  = lstm_spad29_a_data_in;
            spad30_a_data_in = lstm_spad30_a_data_in;  
            spad31_a_data_in =  lstm_spad31_a_data_in;
            spad32_a_data_in = lstm_spad32_a_data_in; 
            mult_out_round_en = lstm_mult_out_round_en;   
            pe_out_sum_a_final_en  = lstm_sum_a_final_en;  
            pe_out_sum_b_final_en = lstm_sum_b_final_en;        
            spad1_a_data_sram_in = lstm_spad1_a_data_sram_in;
            spad2_a_data_sram_in = lstm_spad2_a_data_sram_in;
            spad3_a_data_sram_in = lstm_spad3_a_data_sram_in;
            spad4_a_data_sram_in = lstm_spad4_a_data_sram_in;
            spad5_a_data_sram_in = lstm_spad5_a_data_sram_in;
            spad6_a_data_sram_in = lstm_spad6_a_data_sram_in;
            spad7_a_data_sram_in = lstm_spad7_a_data_sram_in;
            spad8_a_data_sram_in = lstm_spad8_a_data_sram_in;
            spad9_a_data_sram_in = lstm_spad9_a_data_sram_in;
            spad10_a_data_sram_in = lstm_spad10_a_data_sram_in;
            spad11_a_data_sram_in = lstm_spad11_a_data_sram_in;
            spad12_a_data_sram_in = lstm_spad12_a_data_sram_in;
            spad13_a_data_sram_in = lstm_spad13_a_data_sram_in;
            spad14_a_data_sram_in = lstm_spad14_a_data_sram_in;
            spad15_a_data_sram_in = lstm_spad15_a_data_sram_in;
            spad16_a_data_sram_in = lstm_spad16_a_data_sram_in; 
            spad17_a_data_sram_in = 0;
            spad18_a_data_sram_in = 0;
            spad19_a_data_sram_in = 0;
            spad20_a_data_sram_in = 0;
            spad21_a_data_sram_in = 0;
            spad22_a_data_sram_in = 0;
            spad23_a_data_sram_in = 0;
            spad24_a_data_sram_in = 0;
            spad25_a_data_sram_in = 0;
            spad26_a_data_sram_in = 0;
            spad27_a_data_sram_in = 0;
            spad28_a_data_sram_in = 0;
            spad29_a_data_sram_in = 0;
            spad30_a_data_sram_in = 0;
            spad31_a_data_sram_in = 0;
            spad32_a_data_sram_in = 0;
            // spad_a_addr_we_all = {lstm_spad_a_addr_we_1_16,lstm_spad_a_addr_we_1_16}; // gao 16 pe not used
            spad_a_addr_we  =  lstm_spad_a_addr_we;
            spad_a_we_en_all = {lstm_spad_a_we_en_1_16,lstm_spad_a_we_en_1_16};      // gao 16 pe not used

                        
        end
        else if (seg_state_c == decoder)begin
            spad_w_we_en_2_32 = decoder_spad_w_we_en_2_32;
            
            add_a_crl = decoder_add_a_crl;
            add_b_crl = decoder_add_b_crl;                      // 110: hardmard_p 
            mult_int8_crl_1_16  = 0;// not used
            mult_int8_crl_17_32  = 0;//not_used
            out_bq = 0; // decoder: from each PEs,
            scale = decoder_scale; //
            out_bq2 = 0; // decoder: from each PEs,??
            scale2 = decoder_scale;    //
            spad17_a_data_in =  0;
            spad18_a_data_in = 0;
            spad19_a_data_in = 0;
            spad20_a_data_in = 0;
            spad21_a_data_in = 0;
            spad22_a_data_in  = 0;
            spad23_a_data_in = 0;
            spad24_a_data_in = 0;
            spad25_a_data_in = 0;
            spad26_a_data_in  = 0;
            spad27_a_data_in = 0;
            spad28_a_data_in = 0;
            spad29_a_data_in  = 0;
            spad30_a_data_in = 0;  
            spad31_a_data_in =  0;
            spad32_a_data_in = 0;      
            mult_out_round_en = decoder_mult_out_round_en;     
            pe_out_sum_a_final_en = 0; 
            pe_out_sum_b_final_en = 0;
            spad1_a_data_sram_in = decoder_spad1_a_data_sram_in;
            spad2_a_data_sram_in = decoder_spad2_a_data_sram_in;
            spad3_a_data_sram_in = decoder_spad3_a_data_sram_in;
            spad4_a_data_sram_in = decoder_spad4_a_data_sram_in;
            spad5_a_data_sram_in = decoder_spad5_a_data_sram_in;
            spad6_a_data_sram_in = decoder_spad6_a_data_sram_in;
            spad7_a_data_sram_in = decoder_spad7_a_data_sram_in;
            spad8_a_data_sram_in = decoder_spad8_a_data_sram_in;
            if ((decoder_top_state == dcnn1)|(decoder_top_state == cnn11)|(decoder_top_state == cnn12)|(decoder_top_state == dcnn2)) begin
                spad9_a_data_sram_in = decoder_spad9_a_data_sram_in;
                spad10_a_data_sram_in = decoder_spad10_a_data_sram_in;
                spad11_a_data_sram_in = decoder_spad11_a_data_sram_in;
                spad12_a_data_sram_in = decoder_spad12_a_data_sram_in;
                spad13_a_data_sram_in = decoder_spad13_a_data_sram_in;
                spad14_a_data_sram_in = decoder_spad14_a_data_sram_in;
                spad15_a_data_sram_in = decoder_spad15_a_data_sram_in;
                spad16_a_data_sram_in = decoder_spad16_a_data_sram_in;
            end
            else begin
                spad9_a_data_sram_in = decoder_spad1_a_data_sram_in;
                spad10_a_data_sram_in = decoder_spad2_a_data_sram_in;
                spad11_a_data_sram_in = decoder_spad3_a_data_sram_in;
                spad12_a_data_sram_in = decoder_spad4_a_data_sram_in;
                spad13_a_data_sram_in = decoder_spad5_a_data_sram_in;
                spad14_a_data_sram_in = decoder_spad6_a_data_sram_in;
                spad15_a_data_sram_in = decoder_spad7_a_data_sram_in;
                spad16_a_data_sram_in = decoder_spad8_a_data_sram_in;                
            end
            if ((decoder_top_state == dcnn1)|(decoder_top_state == cnn11)|(decoder_top_state == cnn12)) begin
                spad17_a_data_sram_in = decoder_spad17_a_data_sram_in;
                spad18_a_data_sram_in = decoder_spad18_a_data_sram_in;
                spad19_a_data_sram_in = decoder_spad19_a_data_sram_in;
                spad20_a_data_sram_in = decoder_spad20_a_data_sram_in;
                spad21_a_data_sram_in = decoder_spad21_a_data_sram_in;
                spad22_a_data_sram_in = decoder_spad22_a_data_sram_in;
                spad23_a_data_sram_in = decoder_spad23_a_data_sram_in;
                spad24_a_data_sram_in = decoder_spad24_a_data_sram_in;
                spad25_a_data_sram_in = decoder_spad25_a_data_sram_in;
                spad26_a_data_sram_in = decoder_spad26_a_data_sram_in;
                spad27_a_data_sram_in = decoder_spad27_a_data_sram_in;
                spad28_a_data_sram_in = decoder_spad28_a_data_sram_in;
                spad29_a_data_sram_in = decoder_spad29_a_data_sram_in;
                spad30_a_data_sram_in = decoder_spad30_a_data_sram_in;
                spad31_a_data_sram_in = decoder_spad31_a_data_sram_in;
                spad32_a_data_sram_in = decoder_spad32_a_data_sram_in;
            end
            else if (decoder_top_state == dcnn2) begin
                spad17_a_data_sram_in = decoder_spad1_a_data_sram_in;
                spad18_a_data_sram_in = decoder_spad2_a_data_sram_in;
                spad19_a_data_sram_in = decoder_spad3_a_data_sram_in;
                spad20_a_data_sram_in = decoder_spad4_a_data_sram_in;
                spad21_a_data_sram_in = decoder_spad5_a_data_sram_in;
                spad22_a_data_sram_in = decoder_spad6_a_data_sram_in;
                spad23_a_data_sram_in = decoder_spad7_a_data_sram_in;
                spad24_a_data_sram_in = decoder_spad8_a_data_sram_in;
                spad25_a_data_sram_in = decoder_spad9_a_data_sram_in;
                spad26_a_data_sram_in = decoder_spad10_a_data_sram_in;
                spad27_a_data_sram_in = decoder_spad11_a_data_sram_in;
                spad28_a_data_sram_in = decoder_spad12_a_data_sram_in;
                spad29_a_data_sram_in = decoder_spad13_a_data_sram_in;
                spad30_a_data_sram_in = decoder_spad14_a_data_sram_in;
                spad31_a_data_sram_in = decoder_spad15_a_data_sram_in;
                spad32_a_data_sram_in = decoder_spad16_a_data_sram_in;                
            end
            else begin
                spad17_a_data_sram_in = decoder_spad1_a_data_sram_in;
                spad18_a_data_sram_in = decoder_spad2_a_data_sram_in;
                spad19_a_data_sram_in = decoder_spad3_a_data_sram_in;
                spad20_a_data_sram_in = decoder_spad4_a_data_sram_in;
                spad21_a_data_sram_in = decoder_spad5_a_data_sram_in;
                spad22_a_data_sram_in = decoder_spad6_a_data_sram_in;
                spad23_a_data_sram_in = decoder_spad7_a_data_sram_in;
                spad24_a_data_sram_in = decoder_spad8_a_data_sram_in;
                spad25_a_data_sram_in = decoder_spad1_a_data_sram_in;
                spad26_a_data_sram_in = decoder_spad2_a_data_sram_in;
                spad27_a_data_sram_in = decoder_spad3_a_data_sram_in;
                spad28_a_data_sram_in = decoder_spad4_a_data_sram_in;
                spad29_a_data_sram_in = decoder_spad5_a_data_sram_in;
                spad30_a_data_sram_in = decoder_spad6_a_data_sram_in;
                spad31_a_data_sram_in = decoder_spad7_a_data_sram_in;
                spad32_a_data_sram_in = decoder_spad8_a_data_sram_in;                   
            end
            spad_a_addr_we  =  decoder_spad_a_addr_we;
            spad_a_we_en_all = decoder_spad_a_we_en_1_32; 
        end
        else begin
            spad_w_we_en_2_32 =  0;
            add_a_crl = 0;
            add_b_crl = 0;                      // 110: hardmard_p 
            mult_int8_crl_1_16  = 0;
            mult_int8_crl_17_32  = 0;
            out_bq = 0; // lstm: from each PEs
            scale = 0;
            out_bq2 = 0; // lstm: from each PEs
            scale2 = 0;  
            spad17_a_data_in =  0;
            spad18_a_data_in = 0;
            spad19_a_data_in = 0;
            spad20_a_data_in = 0;
            spad21_a_data_in = 0;
            spad22_a_data_in  = 0;
            spad23_a_data_in = 0;
            spad24_a_data_in = 0;
            spad25_a_data_in = 0;
            spad26_a_data_in  = 0;
            spad27_a_data_in = 0;
            spad28_a_data_in = 0;
            spad29_a_data_in  = 0;
            spad30_a_data_in = 0;  
            spad31_a_data_in =  0;
            spad32_a_data_in = 0;    
            mult_out_round_en = 0;   
            pe_out_sum_a_final_en = 0;  
            pe_out_sum_b_final_en = 0;
            spad1_a_data_sram_in = 0;
            spad2_a_data_sram_in = 0;
            spad3_a_data_sram_in = 0;
            spad4_a_data_sram_in = 0;
            spad5_a_data_sram_in = 0;
            spad6_a_data_sram_in = 0;
            spad7_a_data_sram_in = 0;
            spad8_a_data_sram_in = 0;
            spad9_a_data_sram_in = 0;
            spad10_a_data_sram_in = 0;
            spad11_a_data_sram_in = 0;
            spad12_a_data_sram_in = 0;
            spad13_a_data_sram_in = 0;
            spad14_a_data_sram_in = 0;
            spad15_a_data_sram_in = 0;
            spad16_a_data_sram_in = 0; 
            spad17_a_data_sram_in = 0;
            spad18_a_data_sram_in = 0;
            spad19_a_data_sram_in = 0;
            spad20_a_data_sram_in = 0;
            spad21_a_data_sram_in = 0;
            spad22_a_data_sram_in = 0;
            spad23_a_data_sram_in = 0;
            spad24_a_data_sram_in = 0;
            spad25_a_data_sram_in = 0;
            spad26_a_data_sram_in = 0;
            spad27_a_data_sram_in = 0;
            spad28_a_data_sram_in = 0;
            spad29_a_data_sram_in = 0;
            spad30_a_data_sram_in = 0;
            spad31_a_data_sram_in = 0;
            spad32_a_data_sram_in = 0;
            spad_a_addr_we  =  0;
            spad_a_we_en_all = 0; 
        end
    end   
    // control signals: encoder_rdy
    wire encoder_rdy;
    // assign encoder_rdy = (seg_state_c == encoder)? 1:0;
    wire lstm_rdy; 
    wire decoder_rdy;

    assign encoder_rdy = network_rdy;    //change
    assign   lstm_rdy  =  encoder_done; //change
    // assign encoder_rdy = 0;    //change
    // assign   lstm_rdy  =  network_rdy; //change
      
    assign decoder_rdy = lstm_done;// need to be changed later
    
    // assign encoder_rdy = 0;    //change
    // assign   lstm_rdy  =  0; //change
    // assign decoder_rdy = network_rdy;// need to be changed later



    // HL55LPHDSP8192x8B1M16W1SA10	 SRAM_8192x8_act (
    // .Q    (sram_act_dout) ,
    // .CLK   (sclk) ,
    // .ME  (sram_act_en) ,
    // .WE  (sram_act_we) ,
    // .TEST1  (1'b0) , // disable
    // .RME  (1'b0) ,// disable
    // .ADR  (addr_sram_act) ,
    // .WEM(8'B1111_1111),
    // .D    (sram_act_din),
    // .RM(4'B0000));
    
    ENCODER # (
    .ENCODER_WB_DW (ENCODER_WB_DW),
    .SRAM8192_AW(SRAM8192_AW),
    .SRAM8_DW(SRAM8_DW),
    .DATA_DW (INPUT_DW),
    .SRAM_AW (SRAM1024_AW),
    .SRAM_DW (SRAM32_DW), // only sram 1 is used
    .DATA_OUT_DW (DATA_OUT_DW),
    .SPAD_DEPTH(SPAD_DEPTH),
    .SCALE_DW(ENCODER_SCALE_DW),
    .STRIDE (ENCODER_STRIDE),
    .LENGTH_IN(ENCODER_LENGTH_IN),
    .PADDING_PRE (ENCODER_PADDING_PRE),
    .PADDING_POST (ENCODER_PADDING_POST),
    .KERNEL_SIZE (ENCODER_KS),
    .CHANNEL_IN  (ENCODER_CHIN),
    .CHANNEL_OUT (ENCODER_CHOUT),
    .LENGTH_OUT (ENCODER_LENGTH_OUT),
    .ADDR_ENCODER_SRAM_ACT_INIT(ADDR_ENCODER_SRAM_ACT_INIT)
    )
    ENCODER_u (
    .wclk(wclk),
    .sclk(sclk),
    .rst_n(rst_n),
    .seg_state(seg_state),
    //input/output(global) buffer
    .input_signal(input_signal),
    .shift_en(encoder_shift_en),
    
    // communication with network.v
    .addr_encoder_w_init(addr_encoder_w_init), // from top.v
    .addr_encoder_b_init(addr_encoder_b_init), // from top.v
    .addr_encoder_output_scale(addr_encoder_output_scale),// from top.v
    .sram_dout(sram1_dout),
    .encoder_rdy(encoder_rdy),
    .addr_encoder_wb(addr_encoder_wb),  // data width of weight and bias are the same, so no need to differenciate
    .encoder_done(encoder_done),   // encoder completed
    .sram1_en(encoder_sram1_en),
    // communication with SPAD_W
    .spad_w_we_en(encoder_spad1_w_we_en),
    .spad_w_addr_re(encoder_spad_w_addr_re),
    .spad_w_addr_we(encoder_spad_w_addr_we),
    
    //communication with SPRD_A
    .spad_a_addr_re(encoder_spad_a_addr_re),
    .spad_a_data_in(encoder_spad1_a_data_in),
    
    .encoder_b(encoder_b),
    .encoder_w(encoder_w),
    .mult_a_crl(encoder_mult_a_crl),
    .scale(encoder_scale),
    .relu_en(encoder_relu_en),
    .round_en(encoder_round_en),
    .encoder_out_vld(encoder_out_vld),
    .encoder_out(encoder_out),
    .addr_encoder_sram_act(addr_encoder_sram_act),
    .encoder_sram_act_din(encoder_sram_act_din),
    .encoder_sram_act_en(encoder_sram_act_en),
    .encoder_sram_act_we(encoder_sram_act_we)
    );
    
    LSTM #(
    .DATA_DW (DATA_OUT_DW),
    .INPUT_DW(INPUT_DW),
    .DATA_BQ_DW(DATA_BQ_DW),
    .WU_DW (LSTM_WU_DW),
    .B_DW(LSTM_B_DW),
    .SCALE_DW (LSTM_SCALE_DW),
    .NUM_LAYERS(LSTM_NUM_LAYERS),
    .NUM_DIRECTIONS(LSTM_NUM_DIR),
    .HS (LSTM_HS),
    .INPUT_SIZE (ENCODER_CHOUT),
    .SRAM1024_AW (SRAM1024_AW),
    .SRAM8192_AW (SRAM8192_AW),
    .SRAM8_DW(SRAM8_DW),
    .SRAM32_DW (SRAM32_DW),
    .SRAM16_DW (SRAM16_DW),
    .SPAD_DEPTH (SPAD_DEPTH),  
    .PE_NUM (PE_NUM), 
    .T (ENCODER_LENGTH_OUT),
    .ADDR_ENCODER_SRAM_ACT_INIT(ADDR_ENCODER_SRAM_ACT_INIT),//READ ENCODER OUTPUT FROM SRAM
    .ADDR_LSTM10_SRAM_ACT_INIT(ADDR_LSTM10_SRAM_ACT_INIT),
    .ADDR_LSTM11_SRAM_ACT_INIT(ADDR_LSTM11_SRAM_ACT_INIT))
    LSTM_u(
    .wclk(wclk),
    .sclk(sclk),
    .rst_n(rst_n),
    // .act_sr2(act_sr2),
    // .act_sr1(act_sr1),
    // .act_sr3(act_sr3),
    .addr_lstm_w00_init(addr_lstm_w00_init),    // from top.v, w0, u0
    .addr_lstm_u00_init(addr_lstm_u00_init),    // from top.v, w0, u0
    .addr_lstm_b00_init(addr_lstm_b00_init),    // from top.v, b0, b0_reverse, b1, b1_reverse
    .addr_lstm_scales_00_init(addr_lstm_scales_00_init),  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale
    .addr_lstm_w01_init(addr_lstm_w01_init),    // from top.v, w0, u0
    .addr_lstm_u01_init(addr_lstm_u01_init),    // from top.v, w0, u0
    .addr_lstm_b01_init(addr_lstm_b01_init),    // from top.v, b0, b0_reverse, b1, b1_reverse
    .addr_lstm_scales_01_init(addr_lstm_scales_01_init), 
    .addr_lstm_w10_init(addr_lstm_w10_init),    // from top.v, w0, u0
    .addr_lstm_u10_init(addr_lstm_u10_init),    // from top.v, w0, u0
    .addr_lstm_b10_init(addr_lstm_b10_init),    // from top.v, b0, b0_reverse, b1, b1_reverse
    .addr_lstm_scales_10_init(addr_lstm_scales_10_init), 
    .addr_lstm_w11_init(addr_lstm_w11_init),    // from top.v, w0, u0
    .addr_lstm_u11_init(addr_lstm_u11_init),    // from top.v, w0, u0
    .addr_lstm_b11_init(addr_lstm_b11_init),    // from top.v, b0, b0_reverse, b1, b1_reverse
    .addr_lstm_scales_11_init(addr_lstm_scales_11_init), 
    .sram1_dout(sram1_dout),
    .sram2_dout(sram2_dout),
    .sram3_dout(sram3_dout),
    .sram4_dout(sram4_dout),

    .lstm_rdy(lstm_rdy), //top
    .lstm_top_state(lstm_top_state),


    // .lstm_hidden_cat(lstm_hidden_cat), //ht
    // .lstm_hidden_unit_vld(lstm_hidden_unit_vld),
    .addr_sram(addr_sram_lstm),    // data width of weight and bias are the same, so no need to differenciate
    .sram1_en(lstm_sram1_en),
    .sram2_en(lstm_sram2_en),
    .sram3_en(lstm_sram3_en),
    .sram4_en(lstm_sram4_en),

    .lstm_wu(lstm_wu), //segment
    .lstm_done(lstm_done),         // lstm completed
    // .xt_shift_en(lstm_xt_shift_en), //segment
    .spad1_w_we_en(lstm_spad1_w_we_en), // the enable signal for the first spad
    .spad_w_we_en_2_32(lstm_spad_w_we_en_2_32), //the enable signal for the rest of PEs
    .spad_w_addr_re(lstm_spad_w_addr_re), //paralell, the same
    .spad_w_addr_we(lstm_spad_w_addr_we),  // serial operation, so share one addr_we
    .spad_a_addr_re(lstm_spad_a_addr_re), //paralell, the same
    // .spad_a_addr_we_1_16(lstm_spad_a_addr_we_1_16),
    .spad_a_addr_we(lstm_spad_a_addr_we),
    .spad_a_we_en_1_16(lstm_spad_a_we_en_1_16),
    .spad1_a_data_sram_in(lstm_spad1_a_data_sram_in),
    .spad2_a_data_sram_in(lstm_spad2_a_data_sram_in),
    .spad3_a_data_sram_in(lstm_spad3_a_data_sram_in),
    .spad4_a_data_sram_in(lstm_spad4_a_data_sram_in),
    .spad5_a_data_sram_in(lstm_spad5_a_data_sram_in),
    .spad6_a_data_sram_in(lstm_spad6_a_data_sram_in),
    .spad7_a_data_sram_in(lstm_spad7_a_data_sram_in),
    .spad8_a_data_sram_in(lstm_spad8_a_data_sram_in),
    .spad9_a_data_sram_in(lstm_spad9_a_data_sram_in),
    .spad10_a_data_sram_in(lstm_spad10_a_data_sram_in),
    .spad11_a_data_sram_in(lstm_spad11_a_data_sram_in),
    .spad12_a_data_sram_in(lstm_spad12_a_data_sram_in),
    .spad13_a_data_sram_in(lstm_spad13_a_data_sram_in),
    .spad14_a_data_sram_in(lstm_spad14_a_data_sram_in),
    .spad15_a_data_sram_in(lstm_spad15_a_data_sram_in),
    .spad16_a_data_sram_in(lstm_spad16_a_data_sram_in),
    .spad17_a_data_in(lstm_spad17_a_data_in),
    .spad18_a_data_in(lstm_spad18_a_data_in), 
    .spad19_a_data_in(lstm_spad19_a_data_in),
    .spad20_a_data_in(lstm_spad20_a_data_in),  
    .spad21_a_data_in(lstm_spad21_a_data_in),
    .spad22_a_data_in(lstm_spad22_a_data_in), 
    .spad23_a_data_in(lstm_spad23_a_data_in),
    .spad24_a_data_in(lstm_spad24_a_data_in), 
    .spad25_a_data_in(lstm_spad25_a_data_in),
    .spad26_a_data_in(lstm_spad26_a_data_in), 
    .spad27_a_data_in(lstm_spad27_a_data_in),
    .spad28_a_data_in(lstm_spad28_a_data_in), 
    .spad29_a_data_in(lstm_spad29_a_data_in),
    .spad30_a_data_in(lstm_spad30_a_data_in),
    .spad31_a_data_in(lstm_spad31_a_data_in),
    .spad32_a_data_in(lstm_spad32_a_data_in),     
    .pe_out_32b_all(pe_out_32b_all),
    .pe_out_a(pe_out_a),
    .pe_out_b(pe_out_b),
    .mult_a_out_round(mult_a_out_round),
    .mult_b_out_round(mult_b_out_round),    
    .pe_out_sum_a_final(pe_out_sum_a_final),
    .pe_out_sum_b_final(pe_out_sum_b_final),    
    .hardmard_a_all(hardmard_a_all),
    .hardmard_b_all(hardmard_b_all),
    .lstm_b(lstm_b),
    .out_bq(lstm_out_bq), // lstm: from each PEs
    .scale(lstm_scale),
    .out_bq2(lstm_out_bq2), // lstm: from each PEs
    .scale2(lstm_scale2),
    .lstm_ct_temp_out_cat(lstm_ct_temp_out_cat),

    .mult_a_crl(lstm_mult_a_crl), // 00:idle, 01:requantize     
    .mult_b_crl(lstm_mult_b_crl),
    .add_a_crl(lstm_add_a_crl),
    .add_b_crl(lstm_add_b_crl),     
    .mult_int8_crl_1_16(lstm_mult_int8_crl_1_16),
    .mult_int8_crl_17_32(lstm_mult_int8_crl_17_32),
    .mult_out_round_en(lstm_mult_out_round_en),
    .pe_out_sum_a_final_en(lstm_sum_a_final_en),
    .pe_out_sum_b_final_en(lstm_sum_b_final_en),
    .sram_act_dout(sram_act_dout),
    .addr_sram_act(addr_lstm_sram_act),
    .sram_act_en(lstm_sram_act_en),
    .sram_act_we(lstm_sram_act_we),
    .sram_act_din(lstm_sram_act_din));
    
    DECODER#(
    .DATA_DW(DATA_OUT_DW),
    .INPUT_DW(INPUT_DW),
    .DATA_BQ_DW(DATA_BQ_DW),
    .W_DW(DECODER_W_DW),
    .B_DW(DECODER_B_DW),
    .SCALE_DW(DECODER_SCALE_DW),
    .SRAM1024_AW(SRAM1024_AW),
    .SRAM8192_AW(SRAM8192_AW),
    .SRAM512_AW(SRAM512_AW),
    .SRAM8_DW(SRAM8_DW),
    .SRAM32_DW(SRAM32_DW),
    .SRAM16_DW(SRAM16_DW),
    .SPAD_DEPTH(SPAD_DEPTH),
    .PE_NUM(PE_NUM),
    .DCNN1_CHIN(LSTM_NUM_DIR * LSTM_HS ),
    .DCNN1_LENGTH_IN(ENCODER_LENGTH_OUT),
    .DCNN1_CHOUT(DCNN1_CHOUT),
    .DCNN_PADDING(DCNN_PADDING),
    .DCNN_STRIDE(DCNN_STRIDE),
    .DCNN_KS(DCNN_KS),
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
    DECODER_U(
    .wclk(wclk),
    .sclk(sclk),
    .rst_n(rst_n),
    // .act_sr2(act_sr2),
    .act_sr1_1(act_sr1_1),
    // .act_sr3(act_sr3),
    // .act_sr4(act_sr4),
    .decoder_top_state(decoder_top_state),
    .addr_dcnn1_w_init(addr_dcnn1_w_init),
    .addr_dcnn1_scales_init(addr_dcnn1_scales_init),
    .addr_cnn11_w_init(addr_cnn11_w_init),
    .addr_cnn11_scales_init(addr_cnn11_scales_init),
    .addr_cnn11_b_init(addr_cnn11_b_init),
    .addr_cnn12_w_init(addr_cnn12_w_init),
    .addr_cnn12_scales_init(addr_cnn12_scales_init),
    .addr_cnn12_b_init(addr_cnn12_b_init),
    .addr_dcnn2_w_init(addr_dcnn2_w_init),
    .addr_dcnn2_scales_init(addr_dcnn2_scales_init),
    .addr_cnn21_w_init(addr_cnn21_w_init),
    .addr_cnn21_scales_init(addr_cnn21_scales_init),
    .addr_cnn21_b_init(addr_cnn21_b_init),
    .addr_cnn22_w_init(addr_cnn22_w_init),
    .addr_cnn22_scales_init(addr_cnn22_scales_init),
    .addr_cnn22_b_init(addr_cnn22_b_init),

    .sram1_dout(sram1_dout),
    .sram4_dout(sram4_dout),
    .sram5_dout(sram5_dout),
    .sram6_dout(sram6_dout),
    .decoder_rdy(decoder_rdy),

    .mult_a_out_round(mult_a_out_round),
    .mult_b_out_round(mult_b_out_round),    
    .pe_out_sum_a_final(pe_out_sum_a_final),
    .pe_out_sum_b_final(pe_out_sum_b_final), 
    .decoder_out_vld(decoder_out_vld),
    .decoder_done(decoder_done),
    .addr_sram(addr_sram_decoder),

    // .shift_crl_all(shift_crl_all),
    // .cnt_bt_all(cnt_bt_all),
    // .cnt_cho_32(cnt_cho_32),
    .cnn22_is_first(cnn22_is_first),
    .cnn22_is_first_2d(cnn22_is_first_2d),
    .sram1_en(decoder_sram1_en),
    .sram4_en(decoder_sram4_en),
    .sram5_en(decoder_sram5_en),
    .sram6_en(decoder_sram6_en),
    .decoder_w(decoder_w),
    .decoder_scale(decoder_scale),
    .decoder_b1(decoder_b1),
    .decoder_b2(decoder_b2),
    // .decoder_b3(decoder_b3),
    // .decoder_b4(decoder_b4),
    .softmax_out(softmax_out),
    // .decoder_out(decoder_out),
    // .decoder_out_cat(decoder_out_cat),
    // .reorder_en(reorder_en),
    .spad1_w_we_en(decoder_spad1_w_we_en),
    .spad_w_we_en_2_32(decoder_spad_w_we_en_2_32),
    .spad_w_addr_re(decoder_spad_w_addr_re),
    .spad_w_addr_we(decoder_spad_w_addr_we),
    .spad_a_addr_re(decoder_spad_a_addr_re),
    .spad_a_addr_we(decoder_spad_a_addr_we),
    .spad_a_we_en_1_32(decoder_spad_a_we_en_1_32),
    .spad1_a_data_sram_in(decoder_spad1_a_data_sram_in),
    .spad2_a_data_sram_in(decoder_spad2_a_data_sram_in),
    .spad3_a_data_sram_in(decoder_spad3_a_data_sram_in),
    .spad4_a_data_sram_in(decoder_spad4_a_data_sram_in),
    .spad5_a_data_sram_in(decoder_spad5_a_data_sram_in),
    .spad6_a_data_sram_in(decoder_spad6_a_data_sram_in),
    .spad7_a_data_sram_in(decoder_spad7_a_data_sram_in),
    .spad8_a_data_sram_in(decoder_spad8_a_data_sram_in),
    .spad9_a_data_sram_in(decoder_spad9_a_data_sram_in),
    .spad10_a_data_sram_in(decoder_spad10_a_data_sram_in),
    .spad11_a_data_sram_in(decoder_spad11_a_data_sram_in),
    .spad12_a_data_sram_in(decoder_spad12_a_data_sram_in),
    .spad13_a_data_sram_in(decoder_spad13_a_data_sram_in),
    .spad14_a_data_sram_in(decoder_spad14_a_data_sram_in),
    .spad15_a_data_sram_in(decoder_spad15_a_data_sram_in),
    .spad16_a_data_sram_in(decoder_spad16_a_data_sram_in), 
    .spad17_a_data_sram_in(decoder_spad17_a_data_sram_in),
    .spad18_a_data_sram_in(decoder_spad18_a_data_sram_in),
    .spad19_a_data_sram_in(decoder_spad19_a_data_sram_in),
    .spad20_a_data_sram_in(decoder_spad20_a_data_sram_in),
    .spad21_a_data_sram_in(decoder_spad21_a_data_sram_in),
    .spad22_a_data_sram_in(decoder_spad22_a_data_sram_in),
    .spad23_a_data_sram_in(decoder_spad23_a_data_sram_in),
    .spad24_a_data_sram_in(decoder_spad24_a_data_sram_in),
    .spad25_a_data_sram_in(decoder_spad25_a_data_sram_in),
    .spad26_a_data_sram_in(decoder_spad26_a_data_sram_in), 
    .spad27_a_data_sram_in(decoder_spad27_a_data_sram_in),
    .spad28_a_data_sram_in(decoder_spad28_a_data_sram_in),
    .spad29_a_data_sram_in(decoder_spad29_a_data_sram_in),
    .spad30_a_data_sram_in(decoder_spad30_a_data_sram_in),
    .spad31_a_data_sram_in(decoder_spad31_a_data_sram_in),
    .spad32_a_data_sram_in(decoder_spad32_a_data_sram_in),         
    .dcnn1_temp_value_vld(dcnn1_temp_value_vld),
    .dcnn1_transfer_temp_value_en(dcnn1_transfer_temp_value_en),
    .dcnn1_temp_rst(dcnn1_temp_rst),
    .dcnn1_temp_value_for_1(dcnn1_temp_value_for_1),
    

    .mult_a_crl(decoder_mult_a_crl),
    .mult_b_crl(decoder_mult_b_crl),
    .add_a_crl(decoder_add_a_crl),
    .add_b_crl(decoder_add_b_crl),

    .mult_int8_crl_all(decoder_mult_int8_crl),
    .mult_out_round_en(decoder_mult_out_round_en),

    .sram_act_dout(sram_act_dout),
    .addr_sram_act(addr_decoder_sram_act),
    .sram_act_en(decoder_sram_act_en),
    .sram_act_we(decoder_sram_act_we),
    .sram_act_din(decoder_sram_act_din)
    // .sum_a_final_en(decoder_sum_a_final_en),
    // .sum_b_final_en(decoder_sum_b_final_en)
    // .out_temp_A_final(out_temp_A_final)
    );
endmodule