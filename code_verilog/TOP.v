`timescale  1ns/100ps
module TOP 

           (input sclk,
             input wclk,
             input sck,
             input rst_n,
             input [16 -1 :0] sram_din,
            //  input [8-1:0] sram_act_din_test,

            //  input [INPUT_DW*LENGTH_IN-1:0] input_signal,
             input spi_clk,
             input spi_cs_n,
             input spi_mosi,
             
             input mode, // 0--arr, 1--mi + arr
            //  input [LABEL_DW*LENGTH_IN-1:0] softmax_out_all,
            //  input [DATA_OUT_DW*CNN21_OUT-1:0] cnn21_out,
             input data_init_en,
             input input_finish,
             output reg softmax_rdy,
            //  input [ARR_LABEL_DW-1:0] predict_pre,
            //  input [INTEVAL_DW-1:0] rr_pre,
            //  input [INTEVAL_DW-1:0] rr_post,
            //  input [FEATURE_DIM*INPUT_DW-1:0] feature_matrix,
             output spi_miso,
             output [3-1:0] arr_type,
             output mi_type,
             
            //  output signed [2*DATA_OUT_DW-1:0] seg_out,
            //  output signed [DATA_OUT_DW-1: 0] encoder_out, //pe-main out  seg-network in
            //  output seg_out_vld,
             output feature_done,
             output one_beat_done);

    // seg_network

    localparam SRAM16_DW = 16;
    localparam  SRAM32_DW = 32;
    localparam  SPI_DW = 16;
    localparam  DATA_OUT_DW = 8;
    localparam  INPUT_DW = 13;
    localparam  PARAM_DW = 13;
    localparam  LENGTH_IN = 256;
    localparam  LABEL_DW = 2;
    localparam  ARR_LABEL_DW = 3;
    localparam  INTEVAL_DW  = $clog2(LENGTH_IN+1);
    localparam  NUM_LEADS = 12;
    localparam SRAM8192_AW = 13;
    localparam SRAM1024_AW = 10;
    localparam SRAM512_AW = 9;
    // localparam SRAM512_AW = 8;
    localparam  SRAM8_DW = 8;

    localparam ENCODER_WB_DW = 32;
    localparam ENCODER_SCALE_DW = 32;
    localparam SPAD_DEPTH = 8;
    localparam DATA_BQ_DW = 32;
    localparam LSTM_WU_DW = 8;
    localparam LSTM_B_DW = 32;
    localparam LSTM_SCALE_DW = 32;
    localparam PE_NUM = 32;
    localparam DECODER_W_DW = 8;
    localparam DECODER_B_DW = 32;
    localparam DECODER_SCALE_DW = 32;
    localparam FEATURE_SUM_DW = INPUT_DW + 4;

    localparam dcnn1 =  3'b001; //from sram
    localparam cnn11 = 3'b011;
    localparam cnn12 = 3'b111;
    localparam dcnn2 = 3'b101;
    localparam cnn21 = 3'b110;
    localparam cnn22 = 3'b010; 

    reg  network_rdy; // seg_network in
    
    wire [3:0] seg_state; // seg_network in
    wire [2:0] decoder_top_state;

    wire [SRAM32_DW-1: 0] sram1_dout; // sram out,seg-network in, 
    wire [SRAM16_DW-1: 0] sram2_dout; // sram out,seg-network in, 
    wire [SRAM16_DW-1: 0] sram3_dout; // sram out,seg-network in, 
    wire [SRAM16_DW-1: 0] sram4_dout; // sram out,seg-network in, 
    wire [SRAM16_DW-1: 0] sram5_dout;
    wire [SRAM16_DW-1: 0] sram6_dout;
    wire [SRAM16_DW-1: 0] sram7_dout;
    wire [SRAM16_DW-1: 0] sram8_dout;
    wire [SRAM16_DW-1: 0] sram9_dout;
    wire [SRAM16_DW-1: 0] sram10_dout;
    wire [SRAM16_DW-1: 0] sram11_dout;
    wire [SRAM16_DW-1: 0] sram12_dout;
    
    wire [SRAM32_DW -1 :0] sram1_din;
    wire [SRAM16_DW -1 :0] sram2_din; //full
    wire [SRAM16_DW -1 :0] sram3_din; //full
    wire [SRAM16_DW -1 :0] sram4_din; 
    wire [SRAM16_DW -1 :0] sram5_din;
    wire [SRAM16_DW -1 :0] sram6_din;//full
    wire [SRAM16_DW -1 :0] sram7_din;//full
    wire [SRAM16_DW -1 :0] sram8_din;//full
    wire [SRAM16_DW -1 :0] sram9_din;//full
    wire [SRAM16_DW -1 :0] sram10_din;//full
    wire [SRAM16_DW -1 :0] sram11_din;
    wire [SRAM16_DW -1 :0] sram12_din;

    wire [SRAM8_DW-1:0] sram_act_dout;
    wire [SRAM8_DW-1:0] sram_act_din;
    wire sram_act_en;
    wire sram_act_we;
    wire [SRAM8192_AW - 1:0] addr_sram_act;

    wire [SRAM8192_AW-1: 0] addr_sram_seg; // seg-network out
    wire sram1_en;
    wire sram2_en;
    wire sram3_en;
    wire sram4_en;
    wire sram5_en;
    wire sram6_en;
    wire sram7_en;
    wire sram8_en;
    wire sram9_en;
    wire sram10_en;
    wire sram11_en;
    wire sram12_en;
  

    wire sram1_we;
    wire sram2_we;
    wire sram3_we;
    wire sram4_we;
    wire sram5_we;
    wire sram6_we;
    wire sram7_we;
    wire sram8_we;
    wire sram9_we;
    wire sram10_we;
    wire sram11_we;
    wire sram12_we;
    // wire sram_act_we_test;

    wire [SRAM8192_AW-1: 0] addr_sram; // sram in

    wire decoder_done;
    wire ann_done;
    wire ann_rdy;
    wire feature_rdy;
    // wire feature_done;
    wire post_rdy; //
    wire post_done;

    wire [SRAM1024_AW-1:0] addr_encoder_w_init; // seg-network in
    wire [SRAM1024_AW-1:0] addr_encoder_b_init; // seg-network in
    wire [SRAM1024_AW-1:0] addr_encoder_output_scale; // seg-network in
    wire [SRAM8192_AW-1:0] addr_lstm_w00_init;    // seg-network in
    wire [SRAM8192_AW-1:0] addr_lstm_u00_init;    // seg-network in
    wire [SRAM1024_AW-1:0] addr_lstm_b00_init;    //seg-network in
    wire [SRAM8192_AW-1:0] addr_lstm_w01_init;    // seg-network in
    wire [SRAM8192_AW-1:0] addr_lstm_u01_init;    // seg-network in
    wire [SRAM1024_AW-1:0] addr_lstm_b01_init;    //seg-network in
    wire [SRAM8192_AW-1:0] addr_lstm_w10_init;    // seg-network in
    wire [SRAM8192_AW-1:0] addr_lstm_u10_init;    // seg-network in
    wire [SRAM1024_AW-1:0] addr_lstm_b10_init;    //seg-network in
    wire [SRAM8192_AW-1:0] addr_lstm_w11_init;    // seg-network in
    wire [SRAM8192_AW-1:0] addr_lstm_u11_init;    // seg-network in
    wire [SRAM1024_AW-1:0] addr_lstm_b11_init;    //seg-network in
    wire [SRAM1024_AW-1:0]  addr_lstm_scales_00_init;
    wire [SRAM1024_AW-1:0]  addr_lstm_scales_01_init;
    wire [SRAM1024_AW-1:0]  addr_lstm_scales_10_init;
    wire [SRAM1024_AW-1:0]  addr_lstm_scales_11_init;

    wire [SRAM8192_AW-1:0] addr_dcnn1_w_init;
    wire [SRAM1024_AW-1:0] addr_dcnn1_scales_init;
    wire [SRAM8192_AW-1:0] addr_cnn11_w_init;    // from top.v, w0
    wire [SRAM1024_AW-1:0] addr_cnn11_scales_init;  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale
    wire [SRAM1024_AW-1:0] addr_cnn11_b_init;
    wire [SRAM8192_AW-1:0] addr_cnn12_w_init;    // from top.v, w0
    wire [SRAM1024_AW-1:0] addr_cnn12_scales_init;  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale
    wire [SRAM1024_AW-1:0] addr_cnn12_b_init;     
    wire [SRAM512_AW-1:0] addr_dcnn2_w_init;    // from top.v, w0
    wire [SRAM1024_AW-1:0] addr_dcnn2_scales_init; 
    wire [SRAM8192_AW-1:0] addr_cnn21_w_init;    // from top.v, w0
    wire [SRAM1024_AW-1:0] addr_cnn21_scales_init;  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale
    wire [SRAM1024_AW-1:0] addr_cnn21_b_init;
    wire [SRAM8192_AW-1:0] addr_cnn22_w_init;   // from top.v, w0
    wire [SRAM1024_AW-1:0] addr_cnn22_scales_init;  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale
    wire [SRAM1024_AW-1:0] addr_cnn22_b_init;
    
    wire [SRAM1024_AW-1:0] addr_ann1_w_init;  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale
    wire [SRAM1024_AW-1:0] addr_ann1_b_init;
    wire [SRAM1024_AW-1:0] addr_ann2_w_init;
    wire [SRAM1024_AW-1:0] addr_ann2_b_init;
    
    wire [SRAM8192_AW-1:0] addr_ann1_1_w_init;
    wire [SRAM1024_AW-1:0] addr_ann1_1_b_init;
    wire [SRAM8192_AW-1:0] addr_ann2_1_w_init;
    wire [SRAM1024_AW-1:0] addr_ann2_1_b_init;
    wire [SRAM8192_AW-1:0] addr_ann3_1_w_init;
    wire [SRAM1024_AW-1:0] addr_ann3_1_b_init;
    wire [SRAM8192_AW-1:0] addr_ann4_1_w_init;
    wire [SRAM1024_AW-1:0] addr_ann4_1_b_init;
    wire [SRAM8192_AW-1:0] addr_ann5_1_w_init;
    wire [SRAM1024_AW-1:0] addr_ann5_1_b_init;
    wire [SRAM8192_AW-1:0] addr_ann6_1_w_init;
    wire [SRAM1024_AW-1:0] addr_ann6_1_b_init;
    wire [SRAM8192_AW-1:0] addr_ann7_1_w_init;
    wire [SRAM1024_AW-1:0] addr_ann7_1_b_init;
    wire [SRAM8192_AW-1:0] addr_ann8_1_w_init;
    wire [SRAM1024_AW-1:0] addr_ann8_1_b_init;
    wire [SRAM8192_AW-1:0] addr_ann9_1_w_init;
    wire [SRAM1024_AW-1:0] addr_ann9_1_b_init;
    wire [SRAM8192_AW-1:0] addr_ann10_1_w_init;
    wire [SRAM1024_AW-1:0] addr_ann10_1_b_init;
    wire [SRAM8192_AW-1:0] addr_ann11_1_w_init;
    wire [SRAM1024_AW-1:0] addr_ann11_1_b_init;
    wire [SRAM8192_AW-1:0] addr_ann12_1_w_init;
    wire [SRAM1024_AW-1:0] addr_ann12_1_b_init;
    wire [SRAM1024_AW-1:0] addr_ann1_2_w_init;
    wire [SRAM512_AW-1:0] addr_ann1_2_b_init;
    wire [SRAM1024_AW-1:0] addr_ann2_2_w_init;
    wire [SRAM512_AW-1:0] addr_ann2_2_b_init;
    wire [SRAM1024_AW-1:0] addr_ann3_2_w_init;
    wire [SRAM512_AW-1:0] addr_ann3_2_b_init;
    wire [SRAM1024_AW-1:0] addr_ann4_2_w_init;
    wire [SRAM512_AW-1:0] addr_ann4_2_b_init;
    wire [SRAM1024_AW-1:0] addr_ann5_2_w_init;
    wire [SRAM512_AW-1:0] addr_ann5_2_b_init;
    wire [SRAM1024_AW-1:0] addr_ann6_2_w_init;
    wire [SRAM512_AW-1:0] addr_ann6_2_b_init;
    wire [SRAM1024_AW-1:0] addr_ann7_2_w_init;
    wire [SRAM512_AW-1:0] addr_ann7_2_b_init;
    wire [SRAM1024_AW-1:0] addr_ann8_2_w_init;
    wire [SRAM512_AW-1:0] addr_ann8_2_b_init;
    wire [SRAM1024_AW-1:0] addr_ann9_2_w_init;
    wire [SRAM512_AW-1:0] addr_ann9_2_b_init;
    wire [SRAM1024_AW-1:0] addr_ann10_2_w_init;
    wire [SRAM512_AW-1:0] addr_ann10_2_b_init;
    wire [SRAM512_AW-1:0] addr_ann11_2_w_init;
    wire [SRAM512_AW-1:0] addr_ann11_2_b_init;
    wire [SRAM512_AW-1:0] addr_ann12_2_w_init;
    wire [SRAM512_AW-1:0] addr_ann12_2_b_init;

    wire seg_sram1_en ;
    wire seg_sram2_en ;
    wire seg_sram3_en ;
    wire seg_sram4_en ;
    wire seg_sram5_en ;
    wire seg_sram6_en ;
    wire ann_sram7_en ;
    wire ann_sram8_en ;
    wire ann_sram9_en ;
    wire ann_sram10_en ;
    wire ann_sram11_en ;
    wire ann_sram12_en ;





    wire  spad1_w_we_en; //seg-network out,
    wire  spad1_w_we_en_seg; //seg-network out,
    wire  spad1_w_we_en_ann; //ann out
    wire [PE_NUM-2:0] spad_w_we_en_2_32;//seg-network out,

    wire  [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_re_seg; //seg-network out, spad-w in
    wire  [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_re_ann; //seg-network out, spad-w in

    wire  [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_we; // seg-network out, spad-w in
    wire  [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_we_seg; // seg-network out, spad-w in
    wire  [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_we_ann; // seg-network out, spad-w in


    wire  [$clog2(SPAD_DEPTH)-1 : 0] spad_a_addr_re_seg;//seg-network out, spad-a in
    wire  [$clog2(SPAD_DEPTH)-1 : 0] spad_a_addr_re_ann;//seg-network out, spad-a in

    wire [INPUT_DW*SPAD_DEPTH -1 : 0] spad1_a_data_in_seg; //seg-network out, spad-a in
    wire [INPUT_DW*SPAD_DEPTH -1 : 0] spad1_a_data_in; //seg-network out, spad-a in
    wire [INPUT_DW*SPAD_DEPTH -1 : 0] spad1_a_data_in_ann; //seg-network out, spad-a in

    // wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad2_a_data_in; 
    // wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad3_a_data_in;
    // wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad4_a_data_in;
    // wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad5_a_data_in;
    // wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad6_a_data_in;
    // wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad7_a_data_in;
    // wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad8_a_data_in;
    // wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad9_a_data_in;
    // wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad10_a_data_in;
    // wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad11_a_data_in;
    // wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad12_a_data_in;
    // wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad13_a_data_in;
    // wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad14_a_data_in;
    // wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad15_a_data_in;
    // wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad16_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad17_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad18_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad19_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad20_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad21_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad22_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad23_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad24_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad25_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad26_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad27_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad28_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad29_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad30_a_data_in;  
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad31_a_data_in;
    wire [DATA_OUT_DW*SPAD_DEPTH -1 : 0] spad32_a_data_in;
    
    wire signed [ENCODER_WB_DW-1 : 0] encoder_b;  // seg-network out, pe-main input
    wire signed [ENCODER_WB_DW-1 : 0] encoder_w;// seg-network out, spad-w input
    wire signed [ENCODER_SCALE_DW -1 : 0] encoder_scale; //seg-network out, pe-main input
    wire signed [DATA_OUT_DW-1: 0] encoder_out; // pe_main out, seg-network in

    wire signed [LSTM_WU_DW-1 : 0] lstm_wu; //seg-network out, spad-w input
    wire signed [LSTM_B_DW-1: 0] lstm_b;// seg-network out, pe-main input
    
    
    wire signed [DATA_BQ_DW-1: 0] dcnn1_temp_value_for_1;

    wire signed [DECODER_W_DW-1:0] decoder_w; // seg-network out, spad-w input
    wire signed [DECODER_B_DW-1:0] decoder_b1; //seg-network out, pe-main input
    wire signed [DECODER_B_DW-1:0] decoder_b2; //seg-network out, pe-main input

    wire signed [SRAM16_DW-1:0]  ann_w;
    wire signed [SRAM16_DW-1:0]  ann_b;

    wire [PE_NUM*DATA_BQ_DW-1: 0] pe_out_32b_all;  // all the pes out seg-network in
    wire signed [2*DATA_OUT_DW+LSTM_SCALE_DW-1: 0] pe_out_a; // pe-main out, seg-network in
    wire signed [2*DATA_OUT_DW+LSTM_SCALE_DW-1: 0] pe_out_b;// pe-main out, seg-network in
    wire signed [DATA_OUT_DW-1: 0] mult_a_out_round; // pe-main out, seg-network in
    wire signed [DATA_OUT_DW-1: 0] mult_b_out_round;// pe-main out, seg-network in

    wire signed [DATA_OUT_DW-1:0] pe_out_sum_a_final;// pe-main out, seg-network in
    wire signed [DATA_OUT_DW-1:0] pe_out_sum_b_final;// pe-main out, seg-network in
    // wire signed [DATA_OUT_DW-1:0] out_temp_A_final;
    wire signed [PE_NUM * DATA_OUT_DW-1: 0] hardmard_a_all;// seg-network out, pes input
    wire signed [PE_NUM * DATA_OUT_DW-1: 0] hardmard_b_all;// seg-network out, pes input
    
    wire signed [DATA_BQ_DW-1:0] out_bq; //  seg-network out, pe-main input
    wire signed [LSTM_SCALE_DW -1 : 0] scale; //seg-network out, pe-main input
    wire signed [DATA_BQ_DW-1:0] out_bq2; // seg-network out, pe-main input
    wire signed [LSTM_SCALE_DW -1 : 0] scale2; //seg-network out, pe-main input
    // wire signed [2*DATA_OUT_DW-1: 0]  lstm_ct_temp_cat; //seg-network out, pe-main input 
    
    wire mult_out_round_en;
    wire encoder_relu_en;
    wire encoder_round_en;

    wire pe_out_sum_a_final_en;
    wire pe_out_sum_b_final_en;
    wire  cnn22_is_first;
    wire  cnn22_is_first_2d;
    wire [1:0] mult_a_crl_seg;    //seg-network out, pe-main input 
    wire [1:0] mult_a_crl_ann;    //seg-network out, pe-main input 
    wire [1:0] mult_a_crl;  
    wire [1:0] mult_b_crl;//seg-network out, pe-main input
    wire [1:0] add_a_crl;//seg-network out, pe-main input
    wire [1:0] add_b_crl;    //seg-network out, pe-main input                                                                                                                                                                                                                                                                                                                                              , 110: hardmard_p 
    wire [2:0] mult_int8_crl_1_16; //seg-network out, pe-all input
    wire [2:0] mult_int8_crl_17_32; //seg-network out, pe-all input
    wire signed [2*(2*DATA_OUT_DW+LSTM_SCALE_DW)-1: 0] lstm_ct_temp_out_cat;
    
    wire signed [INPUT_DW-1:0] spad1_a_data_out; //spad-a-main out, pe-main in
    wire signed [ENCODER_WB_DW-1:0] spad1_w_data_out;//spad-w-main-out, pe-main in
    
    reg [3*PE_NUM-1:0] mult_int8_crl_all;
    wire [3*PE_NUM-1:0] decoder_mult_int8_crl;
    reg  [$clog2(SPAD_DEPTH)*(PE_NUM-1)-1 : 0] spad_a_addr_re_mem;
    reg  [$clog2(SPAD_DEPTH)*PE_NUM-1 : 0] spad_a_addr_re_all;
    reg  [$clog2(SPAD_DEPTH)*(PE_NUM-1)-1 : 0] spad_w_addr_re_mem;
    reg  [$clog2(SPAD_DEPTH)*PE_NUM-1 : 0] spad_w_addr_re_all;


    // wire signed [2*DATA_OUT_DW-1:0] seg_out;
    // wire seg_out_vld;
    wire init_features_end;

    //TOP FSM
    localparam N           = 4;
    localparam idle        = 4'b0000;
    localparam data_init   = 4'b0001; //write weight and bias into memory
    localparam seg_network = 4'b0010;  //seg-network
    localparam post = 4'b0011;
    localparam feature_map = 4'b0111;
    // localparam input_prepare = 4'b0100;
    localparam ann = 4'b0101;

    localparam done        = 4'b1111;
    reg       [N-1:0]        top_state_c         ; // current state
    reg       [N-1:0]        top_state_n         ; // next state
    
    
    
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            top_state_c <= idle;
        else
            top_state_c <= top_state_n;
    end

    reg first_time;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            first_time <= 1;
        else begin
            if (input_finish) 
                first_time <= 0;
            else 
                first_time <= first_time;
        end
    end  
    reg [$clog2(NUM_LEADS+1)-1 : 0] cnt_lead;
    reg [$clog2(NUM_LEADS+1)-1 : 0] cnt_lead_d;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            cnt_lead <= 0;
        else begin
            if (mode == 1) begin
                if (feature_done) 
                    cnt_lead <= (cnt_lead == NUM_LEADS-1)? 0 : cnt_lead + 1;
                else 
                    cnt_lead <= cnt_lead;
            end
            else cnt_lead <= 0;
        end
    end    

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            cnt_lead_d <= 0;
        else begin
            cnt_lead_d <= cnt_lead;
        end
    end  

    always @(*) begin
        case (top_state_c)
            idle: begin
                if (data_init_en) begin
                    top_state_n = data_init;
                    // if (first_time)
                    //     top_state_n = data_init; //data_init kong yi ge wclk
                    // else
                    //     top_state_n = input_prepare; //need change
                end
                else
                    top_state_n = idle;
            end
            data_init: begin
                if (input_finish) begin
                    if (cnt_lead == 0) top_state_n = seg_network; //need change
                    else top_state_n = post;
                end
                else top_state_n = data_init;
                // if (network_rdy)
                //     top_state_n = seg_network;//need change
                // else
                //     top_state_n = data_init;
            end
            seg_network: begin
                if (decoder_done)
                    top_state_n = post;
                else
                    top_state_n = seg_network;
            end
            post: begin
                if (post_done)
                    top_state_n = feature_map;
                else
                    top_state_n = post;
            end
            feature_map: begin
                if (feature_done) begin
                    if (mode == 0) begin
                        if (init_features_end) top_state_n = ann;
                        else top_state_n = done;
                    end
                    else begin
                        if (cnt_lead_d == NUM_LEADS-1)
                            top_state_n = ann;
                        else 
                            top_state_n = data_init;
                    end
                end
                else
                    top_state_n = feature_map;
            end
            // input_prepare: begin
            //     if (input_finish) begin
            //         if (cnt_lead == 0) top_state_n = seg_network; 
            //         else top_state_n = post;
            //     end
            //     else top_state_n = input_prepare;
            // end
            ann: begin
                if (ann_done)
                    top_state_n = done;
                else
                    top_state_n = ann;                
            end
            done:
            top_state_n         = idle;
            default:top_state_n = idle;
        endcase
    end
    assign one_beat_done = (top_state_c == done)?1:0;

    assign mult_a_crl = (top_state_c == seg_network)?  mult_a_crl_seg : mult_a_crl_ann;
    assign spad_w_addr_we = (top_state_c == seg_network)? spad_w_addr_we_seg:spad_w_addr_we_ann;
    assign spad1_a_data_in = (top_state_c == seg_network)? spad1_a_data_in_seg: spad1_a_data_in_ann;
    assign spad1_w_we_en = (top_state_c == seg_network)? spad1_w_we_en_seg:spad1_w_we_en_ann;

    integer pe_id;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin 
            spad_a_addr_re_mem <= 0;
            spad_w_addr_re_mem <= 0;
        end
        else begin
            if (seg_state == 4'b0100) begin
                spad_a_addr_re_mem[$clog2(SPAD_DEPTH)-1:0] <= spad_a_addr_re_seg;
                spad_w_addr_re_mem[$clog2(SPAD_DEPTH)-1:0] <= spad_w_addr_re_seg;
                for (pe_id = 1; pe_id < PE_NUM-1; pe_id = pe_id+1) begin
                    spad_a_addr_re_mem[$clog2(SPAD_DEPTH)*(pe_id+1)-1-:$clog2(SPAD_DEPTH)] <= spad_a_addr_re_mem[$clog2(SPAD_DEPTH)*pe_id-1-:$clog2(SPAD_DEPTH)];
                    spad_w_addr_re_mem[$clog2(SPAD_DEPTH)*(pe_id+1)-1-:$clog2(SPAD_DEPTH)] <= spad_w_addr_re_mem[$clog2(SPAD_DEPTH)*pe_id-1-:$clog2(SPAD_DEPTH)];
                end
            end
            else begin
                spad_a_addr_re_mem <= 0;
                spad_w_addr_re_mem <= 0;                
            end
        end
    end
    always @(*) begin
        if (top_state_c == seg_network) begin
            if (seg_state == 4'b0001) begin
                mult_int8_crl_all = {PE_NUM{mult_int8_crl_1_16}};
                spad_a_addr_re_all = {PE_NUM{spad_a_addr_re_seg}}; // ????? need to optimize
                spad_w_addr_re_all = {PE_NUM{spad_w_addr_re_seg}};
            end
            else if (seg_state == 4'b0010) begin
                mult_int8_crl_all = {{(PE_NUM/2){mult_int8_crl_17_32}},{(PE_NUM/2){mult_int8_crl_1_16}} };
                spad_a_addr_re_all = {PE_NUM{spad_a_addr_re_seg}};
                spad_w_addr_re_all = {PE_NUM{spad_w_addr_re_seg}};            
            end 
            else if (seg_state == 4'b0100) begin
                mult_int8_crl_all =  decoder_mult_int8_crl;
                if ((decoder_top_state == dcnn1) |(decoder_top_state == cnn11) | (decoder_top_state == cnn12))begin
                    spad_a_addr_re_all = {spad_a_addr_re_mem,spad_a_addr_re_seg};
                    spad_w_addr_re_all = {spad_w_addr_re_mem,spad_w_addr_re_seg};     
                end 
                else if (decoder_top_state == dcnn2) begin
                    spad_a_addr_re_all = {spad_a_addr_re_mem[$clog2(SPAD_DEPTH)*(PE_NUM/2-1)-1:0],spad_a_addr_re_seg,spad_a_addr_re_mem[$clog2(SPAD_DEPTH)*(PE_NUM/2-1)-1:0],spad_a_addr_re_seg};
                    spad_w_addr_re_all = {spad_w_addr_re_mem[$clog2(SPAD_DEPTH)*(PE_NUM/2-1)-1:0],spad_w_addr_re_seg,spad_w_addr_re_mem[$clog2(SPAD_DEPTH)*(PE_NUM/2-1)-1:0],spad_w_addr_re_seg};                 
                end 
                else if ((decoder_top_state == cnn21) | (decoder_top_state == cnn22))   begin
                    spad_a_addr_re_all = {spad_a_addr_re_mem[$clog2(SPAD_DEPTH)*(PE_NUM/4-1)-1:0],spad_a_addr_re_seg,spad_a_addr_re_mem[$clog2(SPAD_DEPTH)*(PE_NUM/4-1)-1:0],spad_a_addr_re_seg,spad_a_addr_re_mem[$clog2(SPAD_DEPTH)*(PE_NUM/4-1)-1:0],spad_a_addr_re_seg,spad_a_addr_re_mem[$clog2(SPAD_DEPTH)*(PE_NUM/4-1)-1:0],spad_a_addr_re_seg};
                    spad_w_addr_re_all = {spad_w_addr_re_mem[$clog2(SPAD_DEPTH)*(PE_NUM/4-1)-1:0],spad_w_addr_re_seg,spad_w_addr_re_mem[$clog2(SPAD_DEPTH)*(PE_NUM/4-1)-1:0],spad_w_addr_re_seg,spad_w_addr_re_mem[$clog2(SPAD_DEPTH)*(PE_NUM/4-1)-1:0],spad_w_addr_re_seg,spad_w_addr_re_mem[$clog2(SPAD_DEPTH)*(PE_NUM/4-1)-1:0],spad_w_addr_re_seg};                 
                end
                else begin
                    spad_a_addr_re_all = 0;
                    spad_w_addr_re_all = 0;                     
                end   
            end
            else begin
                mult_int8_crl_all = 0;
                spad_a_addr_re_all = 0;
                spad_w_addr_re_all = 0;
            end     
        end
        else if (top_state_c == ann) begin
            mult_int8_crl_all = {PE_NUM{2'b00}};
            spad_a_addr_re_all = {{$clog2(SPAD_DEPTH)*(PE_NUM-1) {1'b0}},spad_a_addr_re_ann};
            spad_w_addr_re_all = {{$clog2(SPAD_DEPTH)*(PE_NUM-1) {1'b0}},spad_w_addr_re_ann};            
        end
        else begin
            mult_int8_crl_all = 0;
            spad_a_addr_re_all = 0;
            spad_w_addr_re_all = 0;            
        end
  
    end


    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_1;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_2;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_3;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_4;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_5;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_6;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_7;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_8;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_9;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_10;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_11;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_12;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_13;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_14;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_15;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_16;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_17;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_18;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_19;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_20;    
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_21;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_22;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_23;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_24;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_25;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_26;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_27;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_28;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_29;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_30;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_31;
    wire signed [DATA_BQ_DW-1: 0] psum_out_32b_32;

    reg signed [DATA_BQ_DW-1: 0] psum_out_32b_24_d; // for cnn21, cnn22
    reg signed [DATA_BQ_DW-1: 0] psum_out_32b_32_d; // for cnn21, cnn22
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
                psum_out_32b_24_d <= 0;
                psum_out_32b_32_d <= 0;
            end
        else begin
            psum_out_32b_24_d <= psum_out_32b_24;
            psum_out_32b_32_d <= psum_out_32b_32;            
        end
    end

    assign psum_out_32b_1 = pe_out_32b_all[DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_2 = pe_out_32b_all[2*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_3 = pe_out_32b_all[3*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_4 = pe_out_32b_all[4*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_5 = pe_out_32b_all[5*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_6 = pe_out_32b_all[6*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_7 = pe_out_32b_all[7*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_8 = pe_out_32b_all[8*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_9 = pe_out_32b_all[9*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_10 = pe_out_32b_all[10*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_11 = pe_out_32b_all[11*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_12 = pe_out_32b_all[12*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_13 = pe_out_32b_all[13*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_14 = pe_out_32b_all[14*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_15 = pe_out_32b_all[15*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_16 = pe_out_32b_all[16*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_17 = pe_out_32b_all[17*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_18 = pe_out_32b_all[18*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_19 = pe_out_32b_all[19*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_20 = pe_out_32b_all[20*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_21 = pe_out_32b_all[21*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_22 = pe_out_32b_all[22*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_23 = pe_out_32b_all[23*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_24 = pe_out_32b_all[24*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_25 = pe_out_32b_all[25*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_26 = pe_out_32b_all[26*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_27 = pe_out_32b_all[27*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_28 = pe_out_32b_all[28*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_29 = pe_out_32b_all[29*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_30 = pe_out_32b_all[30*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_31 = pe_out_32b_all[31*DATA_BQ_DW-1-:DATA_BQ_DW];
    assign psum_out_32b_32 = pe_out_32b_all[32*DATA_BQ_DW-1-:DATA_BQ_DW];

    reg  [PE_NUM*DATA_BQ_DW-1: 0] psum_32b_all;
    wire [2:0] lstm_top_state;


    always @(*) begin
        if (seg_state == 4'b0010) begin
            psum_32b_all  = ((lstm_top_state == 3'd1) | (lstm_top_state == 3'd2) )?
                            {psum_out_32b_31, psum_out_32b_30,psum_out_32b_29,psum_out_32b_32,
                            psum_out_32b_27,psum_out_32b_26,psum_out_32b_25,psum_out_32b_28,
                            psum_out_32b_23,psum_out_32b_22,psum_out_32b_21,psum_out_32b_24,
                            psum_out_32b_19,psum_out_32b_18,psum_out_32b_17,psum_out_32b_20,
                            psum_out_32b_15,psum_out_32b_14,psum_out_32b_13,psum_out_32b_16,
                            psum_out_32b_11,psum_out_32b_10,psum_out_32b_9,psum_out_32b_12,
                            psum_out_32b_7,psum_out_32b_6,psum_out_32b_5,psum_out_32b_8,
                            psum_out_32b_3,psum_out_32b_2,psum_out_32b_1,psum_out_32b_4}:
                            {psum_out_32b_31, psum_out_32b_30,psum_out_32b_29,psum_out_32b_32,
                            psum_out_32b_27,psum_out_32b_26,psum_out_32b_25,psum_out_32b_28,
                            psum_out_32b_23,psum_out_32b_22,psum_out_32b_21,psum_out_32b_24,
                            psum_out_32b_19,psum_out_32b_18,psum_out_32b_17,psum_out_32b_20,
                            psum_out_32b_15,psum_out_32b_14,psum_out_32b_13,psum_out_32b_12,
                            psum_out_32b_11,psum_out_32b_10,psum_out_32b_9,psum_out_32b_16,
                            psum_out_32b_7,psum_out_32b_6,psum_out_32b_5,psum_out_32b_4,
                            psum_out_32b_3,psum_out_32b_2,psum_out_32b_1,psum_out_32b_8};
        end 
        else if (seg_state == 4'b0100) begin

            if ((decoder_top_state == dcnn1) |(decoder_top_state == cnn11) | (decoder_top_state == cnn12)) begin
                psum_32b_all = {psum_out_32b_31,psum_out_32b_30,psum_out_32b_29,psum_out_32b_28,
                                psum_out_32b_27,psum_out_32b_26,psum_out_32b_25,psum_out_32b_24,
                                psum_out_32b_23,psum_out_32b_22,psum_out_32b_21,psum_out_32b_20,
                                psum_out_32b_19,psum_out_32b_18,psum_out_32b_17,psum_out_32b_16,
                                psum_out_32b_15,psum_out_32b_14,psum_out_32b_13,psum_out_32b_12,
                                psum_out_32b_11,psum_out_32b_10,psum_out_32b_9,psum_out_32b_8,
                                psum_out_32b_7,psum_out_32b_6,psum_out_32b_5,psum_out_32b_4,
                                psum_out_32b_3,psum_out_32b_2,psum_out_32b_1,psum_out_32b_32};
                end
            else if (decoder_top_state == dcnn2) begin
                psum_32b_all = {psum_out_32b_31,psum_out_32b_30,psum_out_32b_29,psum_out_32b_28,
                                psum_out_32b_27,psum_out_32b_26,psum_out_32b_25,psum_out_32b_24,
                                psum_out_32b_23,psum_out_32b_22,psum_out_32b_21,psum_out_32b_20,
                                psum_out_32b_19,psum_out_32b_18,psum_out_32b_17,{DATA_BQ_DW{1'B0}},
                                psum_out_32b_15,psum_out_32b_14,psum_out_32b_13,psum_out_32b_12,
                                psum_out_32b_11,psum_out_32b_10,psum_out_32b_9,psum_out_32b_8,
                                psum_out_32b_7,psum_out_32b_6,psum_out_32b_5,psum_out_32b_4,
                                psum_out_32b_3,psum_out_32b_2,psum_out_32b_1,{DATA_BQ_DW{1'B0}}};                
            end
            else if ((decoder_top_state == cnn21)|(decoder_top_state == cnn22)) begin
                psum_32b_all = {psum_out_32b_31,psum_out_32b_30,psum_out_32b_29,psum_out_32b_28,
                                psum_out_32b_27,psum_out_32b_26,psum_out_32b_25,{DATA_BQ_DW{1'B0}},
                                psum_out_32b_23,psum_out_32b_22,psum_out_32b_21,psum_out_32b_20,
                                psum_out_32b_19,psum_out_32b_18,psum_out_32b_17,{DATA_BQ_DW{1'B0}},
                                psum_out_32b_15,psum_out_32b_14,psum_out_32b_13,psum_out_32b_12,
                                psum_out_32b_11,psum_out_32b_10,psum_out_32b_9,{DATA_BQ_DW{1'B0}},
                                psum_out_32b_7,psum_out_32b_6,psum_out_32b_5,psum_out_32b_4,
                                psum_out_32b_3,psum_out_32b_2,psum_out_32b_1,{DATA_BQ_DW{1'B0}}};                
            end
            else begin
                psum_32b_all = 0;
            end
        end     
        else begin
            psum_32b_all = 0;
        end     
    end


    wire [(PE_NUM-1)*DATA_OUT_DW*SPAD_DEPTH-1: 0] spad_a_data_in_all; // once
    wire [(PE_NUM-1)*DATA_OUT_DW-1: 0] spad_a_data_sram_in_all; // once
    wire [INPUT_DW-1 : 0] spad1_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad2_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad3_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad4_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad5_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad6_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad7_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad8_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad9_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad10_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad11_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad12_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad13_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad14_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad15_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad16_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad17_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad18_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad19_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad20_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad21_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad22_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad23_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad24_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad25_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad26_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad27_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad28_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad29_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad30_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad31_a_data_sram_in;
    wire [DATA_OUT_DW -1 : 0] spad32_a_data_sram_in;
    wire [(PE_NUM-1)*DATA_OUT_DW-1: 0] spad_a_data_out_all;
    wire [(PE_NUM-1)*DATA_OUT_DW-1: 0] spad_w_data_out_all;
    reg [PE_NUM-1:0] is_sram_in_all;
    // wire  [$clog2(SPAD_DEPTH)*PE_NUM-1 : 0] spad_a_addr_we_all;
    wire  [$clog2(SPAD_DEPTH)-1 : 0] spad_a_addr_we;
    wire  [PE_NUM-1:0] spad_a_we_en_all;

    // assign  spad_a_data_in_all  = {spad32_a_data_in, spad31_a_data_in,spad30_a_data_in,spad29_a_data_in,
    //                         spad28_a_data_in,spad27_a_data_in,spad26_a_data_in,spad25_a_data_in,
    //                         spad24_a_data_in,spad23_a_data_in,spad22_a_data_in,spad21_a_data_in,
    //                         spad20_a_data_in,spad19_a_data_in,spad18_a_data_in,spad17_a_data_in,
    //                         spad16_a_data_in,spad15_a_data_in,spad14_a_data_in,spad13_a_data_in,
    //                         spad12_a_data_in,spad11_a_data_in,spad10_a_data_in,spad9_a_data_in,
    //                         spad8_a_data_in,spad7_a_data_in,spad6_a_data_in,spad5_a_data_in,
    //                         spad4_a_data_in,spad3_a_data_in,spad2_a_data_in};  
    assign  spad_a_data_in_all  = {spad32_a_data_in, spad31_a_data_in,spad30_a_data_in,spad29_a_data_in,
                            spad28_a_data_in,spad27_a_data_in,spad26_a_data_in,spad25_a_data_in,
                            spad24_a_data_in,spad23_a_data_in,spad22_a_data_in,spad21_a_data_in,
                            spad20_a_data_in,spad19_a_data_in,spad18_a_data_in,spad17_a_data_in, {(PE_NUM/2-1)*DATA_OUT_DW*SPAD_DEPTH{1'B0}}};  
    assign  spad_a_data_sram_in_all  = {spad32_a_data_sram_in,spad31_a_data_sram_in,spad30_a_data_sram_in,spad29_a_data_sram_in,
                            spad28_a_data_sram_in,spad27_a_data_sram_in,spad26_a_data_sram_in,spad25_a_data_sram_in,
                            spad24_a_data_sram_in,spad23_a_data_sram_in,spad22_a_data_sram_in,spad21_a_data_sram_in,
                            spad20_a_data_sram_in,spad19_a_data_sram_in,spad18_a_data_sram_in,spad17_a_data_sram_in,
                            spad16_a_data_sram_in,spad15_a_data_sram_in,spad14_a_data_sram_in,spad13_a_data_sram_in,
                            spad12_a_data_sram_in,spad11_a_data_sram_in,spad10_a_data_sram_in,spad9_a_data_sram_in,
                            spad8_a_data_sram_in,spad7_a_data_sram_in,spad6_a_data_sram_in,spad5_a_data_sram_in,
                            spad4_a_data_sram_in,spad3_a_data_sram_in,spad2_a_data_sram_in}; 
    // wire [DATA_OUT_DW -1 : 0] test_sram;
    // assign test_sram = spad_a_data_sram_in_all[DATA_OUT_DW -1 : 0];
    always @(*) begin
        if (top_state_c == seg_network) begin
            if (seg_state == 4'b0001) begin
                is_sram_in_all = {PE_NUM{1'B0}};
            end
            else if (seg_state == 4'b0010) begin
                is_sram_in_all = {{PE_NUM/2{1'B0}}, {PE_NUM/2{1'B1}}};
            end 
            else if (seg_state == 4'b0100) begin
                is_sram_in_all = {PE_NUM{1'B1}};
            end     
            else begin
                is_sram_in_all = {PE_NUM{1'B0}};
            end   
        end
        else if (top_state_c == ann) begin
            is_sram_in_all = {PE_NUM{1'B0}};       
        end
        else begin
            is_sram_in_all = 0;
        end
    end
    reg signed [ENCODER_WB_DW-1:0] spad1_w_data_in; // main
    reg signed [LSTM_WU_DW-1:0] spad_w_data_in;
    always @(*) begin
        
        if (top_state_c == seg_network) begin
            if (seg_state == 4'b0001) begin
                spad1_w_data_in = encoder_w;
                spad_w_data_in = 0; // only one is used
            end
            else if (seg_state == 4'b0010) begin
                spad1_w_data_in = {{(ENCODER_WB_DW - LSTM_WU_DW){lstm_wu[LSTM_WU_DW-1]}},lstm_wu};
                spad_w_data_in = lstm_wu;
            end 
            else if (seg_state == 4'b0100) begin
                spad1_w_data_in = {{(ENCODER_WB_DW - DECODER_W_DW){decoder_w[DECODER_W_DW-1]}},decoder_w};
                spad_w_data_in = decoder_w;
            end     
            else begin
                spad1_w_data_in = 0;
                spad_w_data_in = 0;
            end   
        end
        else if (top_state_c == ann) begin

                spad1_w_data_in = {{(ENCODER_WB_DW - SRAM16_DW){ann_w[SRAM16_DW-1]}},ann_w};
                spad_w_data_in = 0; // only one is used  

         
        end
        else begin
            spad1_w_data_in = 0;
            spad_w_data_in = 0;
        end
    end

    // reg [100:0] count_total;
    // always @(posedge wclk or negedge rst_n) begin
    //     if (!rst_n) count_total <= 0;
    //     else begin
    //         if ((top_state_c == data_init) | (top_state_c == seg_network)) begin
    //             count_total <= count_total + 1;
    //         end
    //         else count_total <= 0;
    //     end
    // end    
    //sram_init, sram0_we,sram1_we sram_en, addr_sram
    // RAM initial dat
    localparam ENCODER_W_DP  = 256 ;  // 8x32 sram 1,  32bit
    localparam ENCODER_B_DP  = 32;  // 32 sram 1, 32bit
    localparam ENCODER_SCALE_DP      = 1 ;  // scale sram 1, 32bit
    localparam LSTM_W00_DP = 2048; //128*32 sram 2, 8bit
    localparam LSTM_U00_DP = 2048; //128*32 sram 2, 8bit
    localparam LSTM_W01_DP = 2048; //128*32 sram 2, 8bit
    localparam LSTM_U01_DP = 2048; //128*32 sram 2, 8bit

    localparam LSTM_W10_DP = 4096; //128*64 sram 3, 8bit
    localparam LSTM_U10_DP = 2048; //128*32 sram 3, 8bit

    localparam LSTM_W11_DP = 4096; //128*64 sram 4, 8bit
    localparam LSTM_U11_DP = 2048; //128*64 sram 3, 8bit   

    localparam LSTM_B_DP = 128; // sram 1, 32bit
    localparam LSTM_SCALES_DP = 7;// sram 1, 32bit

    localparam DECODER_SCALE_DP = 1;// sram 1, 32bit
    localparam DECODER_CNN11_B_DP =  32;
    localparam DECODER_CNN12_B_DP =  16;
    localparam DECODER_CNN21_B_DP =  8;
    localparam DECODER_CNN22_B_DP =  4;

    localparam DECODER_DCNN1_W_DP = 8192; //sram5
    // localparam DECODER_CNN11_W_DP_SRAM3 = 2048; //sram3,2048,
    // localparam DECODER_CNN11_W_DP_SRAM4 = 32*32*5/2 - 2048; //sram4,2560-2048,
    localparam DECODER_CNN11_W_DP = 32*32*5/2; //sram4,2560
    localparam DECODER_CNN12_W_DP = 32*16*5/2; //sram4,1280
    localparam DECODER_DCNN2_W_DP = 16*8*8/2; //sram4，512
    localparam DECODER_CNN21_W_DP = 8*8*5/2; //sram4,160
    localparam DECODER_CNN22_W_DP = 8*4*5/2; //sram4,80

    localparam ANN1_W_DP = 22*32;
    localparam ANN1_B_DP = 32;
    localparam ANN2_W_DP = 32*5;
    localparam ANN2_B_DP = 5;
    localparam ANN1_MI_W_DP = 64*32;
    localparam ANN1_MI_B_DP = 32;
    localparam ANN2_MI_W_DP = 32*2;
    localparam ANN2_MI_B_DP = 2;

    localparam TOTAL_SRAM1_DP = ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP
                                + 4 * LSTM_B_DP+ 4 * LSTM_SCALES_DP
                                + 6 * DECODER_SCALE_DP + DECODER_CNN11_B_DP + DECODER_CNN12_B_DP + DECODER_CNN21_B_DP + DECODER_CNN22_B_DP;

    localparam TOTAL_SRAM2_DP = LSTM_W00_DP + LSTM_U00_DP + LSTM_W01_DP + LSTM_U01_DP;

    localparam TOTAL_SRAM3_DP = LSTM_W10_DP + LSTM_U10_DP + LSTM_U11_DP;

    localparam TOTAL_SRAM4_DP = LSTM_W11_DP + DECODER_CNN11_W_DP + DECODER_CNN12_W_DP + DECODER_CNN21_W_DP + DECODER_CNN22_W_DP;

    localparam TOTAL_SRAM5_DP = DECODER_DCNN1_W_DP;

    localparam TOTAL_SRAM6_DP = DECODER_DCNN2_W_DP;

    localparam TOTAL_SRAM7_DP = ANN1_W_DP + ANN1_B_DP + ANN2_W_DP + ANN2_B_DP;

    localparam TOTAL_SRAM8_DP = ANN1_MI_W_DP  * 4; // ann1_1_w,ann2_1_w,ann3_1_w,ann4_1_w
    localparam TOTAL_SRAM9_DP = ANN1_MI_W_DP  * 4; // ann5_1_w,ann6_1_w,ann7_1_w,ann8_1_w
    localparam TOTAL_SRAM10_DP = ANN1_MI_W_DP  * 4; // ann9_1_w,ann10_1_w,ann11_1_w,ann12_1_w
    
    localparam TOTAL_SRAM11_DP = ANN1_MI_B_DP  * 12 + ANN2_MI_W_DP * 10; // 1024,
    localparam TOTAL_SRAM12_DP = ANN2_MI_W_DP * 2 + ANN2_MI_B_DP*12;


    reg [SRAM8192_AW-1 : 0] addr_sram_init ;
    wire [SRAM8192_AW-1 : 0] addr_sram_ann ;
    // reg [SRAM8192_AW-1 : 0] addr_sram_act_test ;

    
    wire sram1_init_en;
    wire sram2_init_en;
    wire sram3_init_en;
    wire sram4_init_en;
    wire sram5_init_en;
    wire sram6_init_en;
    wire sram7_init_en;
    wire sram8_init_en;
    wire sram9_init_en;
    wire sram10_init_en;
    wire sram11_init_en;
    wire sram12_init_en;
    // wire sram_act_init_en; //change

    reg [4:0] sram_id; //0,1,2,3,4.。。12


    assign sram1_init_en = (sram_id == 1)? 1:0; 
    assign sram2_init_en = (sram_id == 2)? 1:0; 
    assign sram3_init_en = (sram_id == 3)? 1:0; 
    assign sram4_init_en = (sram_id == 4)? 1:0; 
    assign sram5_init_en = (sram_id == 5)? 1:0; 
    assign sram6_init_en = (sram_id == 6)? 1:0; 
    assign sram7_init_en = (sram_id == 7)? 1:0;  
    assign sram8_init_en = (sram_id == 8)? 1:0; 
    assign sram9_init_en = (sram_id == 9)? 1:0; 
    assign sram10_init_en = (sram_id == 10)? 1:0; 
    assign sram11_init_en = (sram_id == 11)? 1:0; 
    assign sram12_init_en = (sram_id == 12)? 1:0;

    // assign sram_act_init_en = ((sram_id == 13)|(sram_id == 14))? 1:0; //change
    // assign sram_act_init_en = (sram_id == 13)? 1:0;

    assign addr_sram  = ( (top_state_c == data_init) & first_time) ? addr_sram_init : (top_state_c == seg_network)?  addr_sram_seg : addr_sram_ann; 

    assign sram1_en    = sram1_init_en | seg_sram1_en;
    assign sram2_en    = sram2_init_en | seg_sram2_en;
    assign sram3_en    = sram3_init_en | seg_sram3_en;
    assign sram4_en    = sram4_init_en | seg_sram4_en;
    assign sram5_en    = sram5_init_en | seg_sram5_en;
    assign sram6_en    = sram6_init_en | seg_sram6_en;
    assign sram7_en    = sram7_init_en | ann_sram7_en;
    assign sram8_en    = sram8_init_en | ann_sram8_en;
    assign sram9_en    = sram9_init_en | ann_sram9_en;
    assign sram10_en    = sram10_init_en | ann_sram10_en;
    assign sram11_en    = sram11_init_en | ann_sram11_en;
    assign sram12_en    = sram12_init_en | ann_sram12_en;

    assign sram1_we    = (sram1_init_en) ? 1: 0; //1 is write, 0 is read
    assign sram2_we    = (sram2_init_en) ? 1: 0; //1 is write, 0 is read
    assign sram3_we    = (sram3_init_en) ? 1: 0; //1 is write, 0 is read
    assign sram4_we    = (sram4_init_en) ? 1: 0; //1 is write, 0 is read
    assign sram5_we    = (sram5_init_en) ? 1: 0;
    assign sram6_we    = (sram6_init_en) ? 1: 0;
    assign sram7_we    = (sram7_init_en) ? 1: 0;
    assign sram8_we    = (sram8_init_en) ? 1: 0;
    assign sram9_we    = (sram9_init_en) ? 1: 0;
    assign sram10_we    = (sram10_init_en) ? 1: 0;
    assign sram11_we    = (sram11_init_en) ? 1: 0;
    assign sram12_we    = (sram12_init_en) ? 1: 0;

    // assign sram_act_we_test    = (sram_act_init_en) ? 1: 0; //change

    reg [SRAM16_DW-1:0] sram_din_d;
    reg is_sram1_lsb;
    always @(negedge sclk or negedge rst_n) begin
        if (!rst_n) sram_din_d <= 0;
        else begin
            if (sram1_init_en)
                sram_din_d <= sram_din;
            else
                sram_din_d <= 0;
        end
    end
    always @(negedge sclk or negedge rst_n) begin
        if (!rst_n) is_sram1_lsb <= 1;
        else begin
            if (sram1_init_en)
                is_sram1_lsb <= ~is_sram1_lsb;
            else
                is_sram1_lsb <= 1;
        end
    end    
    assign sram1_din = (sram1_init_en)? {sram_din, sram_din_d}  : 0;
    assign sram2_din = (sram2_init_en)? sram_din : 0;
    assign sram3_din = (sram3_init_en)? sram_din : 0;
    assign sram4_din = (sram4_init_en)? sram_din : 0;
    assign sram5_din = (sram5_init_en)? sram_din : 0;
    assign sram6_din = (sram6_init_en)? sram_din : 0;
    assign sram7_din = (sram7_init_en)? sram_din : 0;
    assign sram8_din = (sram8_init_en)? sram_din : 0;
    assign sram9_din = (sram9_init_en)? sram_din : 0;
    assign sram10_din = (sram10_init_en)? sram_din : 0;
    assign sram11_din = (sram11_init_en)? sram_din : 0;
    assign sram12_din = (sram12_init_en)? sram_din : 0;




    // reg data_init_en_d;
    // always @(posedge wclk or negedge rst_n) begin
    //     if (!rst_n) data_init_en_d <= 0;
    //     else data_init_en_d <= data_init_en; //data_init_en_d 为1时，top_state_c == data_init， sram_init_en =1
    // end

    always @(negedge sclk or negedge rst_n) begin
        if (!rst_n) begin
            sram_id <= 0;
            addr_sram_init <= 0;
            // addr_sram_act_test <= 0;
        end
        else begin
            if ((top_state_c == data_init) & first_time) begin
                if (sram_id == 0) begin
                    sram_id <= 1;
                    addr_sram_init <= 0;                     
                end
                else if (sram_id == 1) begin
                    if ((addr_sram_init == TOTAL_SRAM1_DP-1)& (!is_sram1_lsb) )begin// end of init sram1
                        sram_id <= sram_id + 1;
                        addr_sram_init <= 0;                        
                    end

                    else begin

                        sram_id <= sram_id;
                        addr_sram_init <= (!is_sram1_lsb)? addr_sram_init + 1 : addr_sram_init;                        
                    end                    
                end
                else if (sram_id == 2) begin
                    if (addr_sram_init == TOTAL_SRAM2_DP-1) begin// end of init sram2
                        sram_id <= sram_id + 1; // end
                        addr_sram_init <= 0;                        
                    end

                    else begin
                        sram_id <= sram_id;
                        addr_sram_init <= addr_sram_init + 1;                        
                    end                    
                end
                else if (sram_id == 3) begin
                    if (addr_sram_init == TOTAL_SRAM3_DP-1) begin// end of init sram3
                        sram_id <= sram_id + 1; 
                        addr_sram_init <= 0;                        
                    end

                    else begin
                        sram_id <= sram_id;
                        addr_sram_init <= addr_sram_init + 1;                        
                    end                    
                end
                else if (sram_id == 4) begin
                    if (addr_sram_init == TOTAL_SRAM4_DP-1) begin// end of init sram4
                        sram_id <= sram_id + 1; 
                        addr_sram_init <= 0;                        
                    end

                    else begin
                        sram_id <= sram_id;
                        addr_sram_init <= addr_sram_init + 1;                        
                    end                    
                end
                else if (sram_id == 5) begin
                    if (addr_sram_init == TOTAL_SRAM5_DP-1) begin// end of init sram5
                        sram_id <= sram_id + 1; // end
                        addr_sram_init <= 0;                        
                    end

                    else begin
                        sram_id <= sram_id;
                        addr_sram_init <= addr_sram_init + 1;                        
                    end                    
                end
                else if (sram_id == 6) begin
                    if (addr_sram_init == TOTAL_SRAM6_DP-1) begin// end of init sram5
                        sram_id <= 7; // end
                        addr_sram_init <= 0;                        
                    end

                    else begin
                        sram_id <= sram_id;
                        addr_sram_init <= addr_sram_init + 1;                        
                    end                    
                end
                else if (sram_id == 7) begin
                    if (addr_sram_init == TOTAL_SRAM7_DP-1) begin// end of init sram5
                        sram_id <= 8; // end
                        addr_sram_init <= 0;                        
                    end

                    else begin
                        sram_id <= sram_id;
                        addr_sram_init <= addr_sram_init + 1;                        
                    end                    
                end
                else if (sram_id == 8) begin
                    if (addr_sram_init == TOTAL_SRAM8_DP-1) begin// end of init sram5
                        sram_id <= 9; // end
                        addr_sram_init <= 0;                        
                    end

                    else begin
                        sram_id <= sram_id;
                        addr_sram_init <= addr_sram_init + 1;                        
                    end                    
                end
                else if (sram_id == 9) begin
                    if (addr_sram_init == TOTAL_SRAM9_DP-1) begin// end of init sram5
                        sram_id <= 10; // end
                        addr_sram_init <= 0;                        
                    end

                    else begin
                        sram_id <= sram_id;
                        addr_sram_init <= addr_sram_init + 1;                        
                    end                    
                end
                else if (sram_id == 10) begin
                    if (addr_sram_init == TOTAL_SRAM10_DP-1) begin// end of init sram5
                        sram_id <= 11; // end
                        addr_sram_init <= 0;                        
                    end

                    else begin
                        sram_id <= sram_id;
                        addr_sram_init <= addr_sram_init + 1;                        
                    end                    
                end
                else if (sram_id == 11) begin
                    if (addr_sram_init == TOTAL_SRAM11_DP-1) begin// end of init sram5
                        sram_id <= 12; 
                        addr_sram_init <= 0;                       
                    end

                    else begin
                        sram_id <= sram_id;
                        addr_sram_init <= addr_sram_init + 1;                        
                    end                    
                end
                else if (sram_id == 12) begin
                    if (addr_sram_init == TOTAL_SRAM12_DP-1) begin// end of init sram5
                        sram_id <= 13; // end
                        addr_sram_init <= addr_sram_init;                        
                    end

                    else begin
                        sram_id <= sram_id;
                        addr_sram_init <= addr_sram_init + 1;                        
                    end                    
                end
                // else if (sram_id == 13) begin
                //     if (addr_sram_act_test == 6144 + 2048  -1) begin// end of init sram5
                //         sram_id <= 14; // end
                //         addr_sram_act_test <= addr_sram_act_test;                        
                //     end

                //     else begin
                //         sram_id <= sram_id;
                //         addr_sram_act_test <= addr_sram_act_test + 1;                        
                //     end                    
                // end     
                // else if (sram_id == 13) begin
                //     if (addr_sram_act_test == 2048  -1) begin// end of init sram5
                //         sram_id <= 14; // end
                //         addr_sram_act_test <= 2048*3;                        
                //     end

                //     else begin
                //         sram_id <= sram_id;
                //         addr_sram_act_test <= addr_sram_act_test + 1;                        
                //     end                      
                // end 
                // else if (sram_id == 14) begin
                //     if (addr_sram_act_test == 2048*4  -1) begin// end of init sram5
                //         sram_id <= 15; // end
                //         addr_sram_act_test <= addr_sram_act_test;                        
                //     end

                //     else begin
                //         sram_id <= sram_id;
                //         addr_sram_act_test <= addr_sram_act_test + 1;                        
                //     end                      
                // end                
                else begin
                    sram_id <= sram_id;
                    addr_sram_init <= addr_sram_init; 
                end
            end
            else begin // return to the start
                sram_id <= 0;
                addr_sram_init <= 0;  
                // addr_sram_act_test <= 6144;   //change                
            end
        end 
    end   

    
    
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) network_rdy <= 0;
        else  begin
            // network_rdy <= ((top_state_c == data_init)&(sram_id == 13) & (addr_sram_init == TOTAL_SRAM12_DP-1))? 1:((top_state_c==done)?1:0) ;
            // network_rdy <= (first_time)? (((top_state_c == data_init)&(sram_id == 13) & (addr_sram_init == TOTAL_SRAM12_DP-1))? 1:0): (((input_finish) & (cnt_lead == 0))?1:0) ;
            network_rdy <=  ((input_finish) & (cnt_lead == 0))?1:0 ;
        end
    end
    

    
    // sram1    
    assign addr_encoder_w_init       = 0;
    assign addr_encoder_b_init       = ENCODER_W_DP; // 32bit
    assign addr_encoder_output_scale = ENCODER_W_DP + ENCODER_B_DP;
    assign addr_lstm_b00_init         = ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP;
    assign  addr_lstm_scales_00_init  = ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + LSTM_B_DP;
    assign addr_lstm_b01_init         = ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + LSTM_B_DP + LSTM_SCALES_DP;
    assign  addr_lstm_scales_01_init  = ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + 2 * LSTM_B_DP + LSTM_SCALES_DP;
    assign addr_lstm_b10_init         = ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + 2 * LSTM_B_DP + 2 * LSTM_SCALES_DP;
    assign  addr_lstm_scales_10_init  = ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + 3 * LSTM_B_DP + 2 * LSTM_SCALES_DP;
    assign addr_lstm_b11_init         = ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + 3 * LSTM_B_DP + 3 * LSTM_SCALES_DP;
    assign  addr_lstm_scales_11_init  = ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + 4 * LSTM_B_DP + 3 * LSTM_SCALES_DP;

    assign addr_dcnn1_scales_init = ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + 4 * LSTM_B_DP + 4 * LSTM_SCALES_DP ;
    assign addr_cnn11_b_init = ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + 4 * LSTM_B_DP + 4 * LSTM_SCALES_DP  + DECODER_SCALE_DP;
    assign addr_cnn11_scales_init = ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + 4 * LSTM_B_DP + 4 * LSTM_SCALES_DP  + DECODER_SCALE_DP + DECODER_CNN11_B_DP;
    assign addr_cnn12_b_init =  ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + 4 * LSTM_B_DP + 4 * LSTM_SCALES_DP  + 2*DECODER_SCALE_DP + DECODER_CNN11_B_DP;
    assign addr_cnn12_scales_init = ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + 4 * LSTM_B_DP + 4 * LSTM_SCALES_DP  + 2*DECODER_SCALE_DP + DECODER_CNN11_B_DP + DECODER_CNN12_B_DP;
    assign addr_dcnn2_scales_init = ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + 4 * LSTM_B_DP + 4 * LSTM_SCALES_DP  + 3*DECODER_SCALE_DP + DECODER_CNN11_B_DP + DECODER_CNN12_B_DP;
    assign addr_cnn21_b_init = ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + 4 * LSTM_B_DP + 4 * LSTM_SCALES_DP  + 4*DECODER_SCALE_DP + DECODER_CNN11_B_DP + DECODER_CNN12_B_DP;
    assign addr_cnn21_scales_init = ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + 4 * LSTM_B_DP + 4 * LSTM_SCALES_DP  + 4*DECODER_SCALE_DP + DECODER_CNN11_B_DP + DECODER_CNN12_B_DP + DECODER_CNN21_B_DP;
    assign addr_cnn22_b_init =  ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + 4 * LSTM_B_DP + 4 * LSTM_SCALES_DP  + 5*DECODER_SCALE_DP + DECODER_CNN11_B_DP + DECODER_CNN12_B_DP + DECODER_CNN21_B_DP;
    assign addr_cnn22_scales_init  =  ENCODER_W_DP + ENCODER_B_DP + ENCODER_SCALE_DP + 4 * LSTM_B_DP + 4 * LSTM_SCALES_DP  + 5*DECODER_SCALE_DP + DECODER_CNN11_B_DP + DECODER_CNN12_B_DP + DECODER_CNN21_B_DP + DECODER_CNN22_B_DP;


    // sram2    
    assign addr_lstm_w00_init         = 0;
    assign addr_lstm_u00_init         = LSTM_W00_DP;         
    assign addr_lstm_w01_init         = LSTM_W00_DP + LSTM_U00_DP;  
    assign addr_lstm_u01_init         = LSTM_W00_DP + LSTM_U00_DP + LSTM_W01_DP;  

    // sram3
    assign addr_lstm_w10_init         = 0;
    assign addr_lstm_u10_init         = LSTM_W10_DP;      
    assign addr_lstm_u11_init         = LSTM_W10_DP + LSTM_U10_DP;   
    // sram4
    assign addr_lstm_w11_init         = 0; 
    assign addr_cnn11_w_init          = LSTM_W11_DP;
    assign addr_cnn12_w_init          = LSTM_W11_DP + DECODER_CNN11_W_DP;    
    assign addr_cnn21_w_init          = LSTM_W11_DP + DECODER_CNN11_W_DP + DECODER_CNN12_W_DP ;        
    assign addr_cnn22_w_init          = LSTM_W11_DP + DECODER_CNN11_W_DP + DECODER_CNN12_W_DP + DECODER_CNN21_W_DP;    

    // sram5
    assign addr_dcnn1_w_init         = 0; 

    //sram6
    assign addr_dcnn2_w_init  = 0;

    //sram7
    assign addr_ann1_w_init  = 0;
    assign addr_ann1_b_init  = ANN1_W_DP;
    assign addr_ann2_w_init  = ANN1_W_DP + ANN1_B_DP;
    assign addr_ann2_b_init  = ANN1_W_DP + ANN1_B_DP + ANN2_W_DP;

    // sram8
    assign addr_ann1_1_w_init  = 0;
    assign addr_ann2_1_w_init  = ANN1_MI_W_DP;
    assign addr_ann3_1_w_init  = 2*ANN1_MI_W_DP;
    assign addr_ann4_1_w_init  = 3*ANN1_MI_W_DP;

    // sram9
    assign addr_ann5_1_w_init  = 0;
    assign addr_ann6_1_w_init  = ANN1_MI_W_DP;
    assign addr_ann7_1_w_init  = 2*ANN1_MI_W_DP;
    assign addr_ann8_1_w_init  = 3*ANN1_MI_W_DP;

    //sram10
    assign addr_ann9_1_w_init  = 0;
    assign addr_ann10_1_w_init  = ANN1_MI_W_DP;
    assign addr_ann11_1_w_init  = 2*ANN1_MI_W_DP;
    assign addr_ann12_1_w_init  = 3*ANN1_MI_W_DP;

    //sram11
    assign addr_ann1_1_b_init = 0;
    assign addr_ann2_1_b_init = ANN1_MI_B_DP;
    assign addr_ann3_1_b_init = 2*ANN1_MI_B_DP;
    assign addr_ann4_1_b_init = 3*ANN1_MI_B_DP;
    assign addr_ann5_1_b_init = 4*ANN1_MI_B_DP;
    assign addr_ann6_1_b_init = 5*ANN1_MI_B_DP;
    assign addr_ann7_1_b_init = 6*ANN1_MI_B_DP;
    assign addr_ann8_1_b_init = 7*ANN1_MI_B_DP;
    assign addr_ann9_1_b_init = 8*ANN1_MI_B_DP;
    assign addr_ann10_1_b_init = 9*ANN1_MI_B_DP;
    assign addr_ann11_1_b_init = 10*ANN1_MI_B_DP;
    assign addr_ann12_1_b_init = 11*ANN1_MI_B_DP;
    assign addr_ann1_2_w_init = 12*ANN1_MI_B_DP;
    assign addr_ann2_2_w_init = 12*ANN1_MI_B_DP + ANN2_MI_W_DP;
    assign addr_ann3_2_w_init = 12*ANN1_MI_B_DP + 2 * ANN2_MI_W_DP;    
    assign addr_ann4_2_w_init = 12*ANN1_MI_B_DP + 3 * ANN2_MI_W_DP;
    assign addr_ann5_2_w_init = 12*ANN1_MI_B_DP + 4 * ANN2_MI_W_DP;
    assign addr_ann6_2_w_init = 12*ANN1_MI_B_DP + 5 * ANN2_MI_W_DP;
    assign addr_ann7_2_w_init = 12*ANN1_MI_B_DP + 6 * ANN2_MI_W_DP;
    assign addr_ann8_2_w_init = 12*ANN1_MI_B_DP + 7 * ANN2_MI_W_DP;
    assign addr_ann9_2_w_init = 12*ANN1_MI_B_DP + 8 * ANN2_MI_W_DP;
    assign addr_ann10_2_w_init = 12*ANN1_MI_B_DP + 9 * ANN2_MI_W_DP;

    //sram 12
    assign addr_ann1_2_b_init = 0;
    assign addr_ann2_2_b_init = ANN2_MI_B_DP;    
    assign addr_ann3_2_b_init = 2 * ANN2_MI_B_DP; 
    assign addr_ann4_2_b_init = 3 * ANN2_MI_B_DP; 
    assign addr_ann5_2_b_init = 4 * ANN2_MI_B_DP; 
    assign addr_ann6_2_b_init = 5 * ANN2_MI_B_DP; 
    assign addr_ann7_2_b_init = 6 * ANN2_MI_B_DP;
    assign addr_ann8_2_b_init = 7 * ANN2_MI_B_DP;  
    assign addr_ann9_2_b_init = 8 * ANN2_MI_B_DP; 
    assign addr_ann10_2_b_init = 9 * ANN2_MI_B_DP; 
    assign addr_ann11_2_b_init = 10 * ANN2_MI_B_DP; 
    assign addr_ann12_2_b_init = 11 * ANN2_MI_B_DP; 
    assign addr_ann11_2_w_init = 12 * ANN2_MI_B_DP;
    assign addr_ann12_2_w_init = 12 * ANN2_MI_B_DP + ANN2_MI_W_DP; 
    


    // localparam ENCODER_PADDING_PRE = 3;
    // localparam ENCODER_PADDING_POST = 1;
    localparam ENCODER_STRIDE = 4;
    localparam DCNN1_LENGTH_OUT = LENGTH_IN/2;
    localparam CNN22_LENGTH_OUT =  LENGTH_IN;  
    localparam INIT_NUM_BEATS = 8; 
    localparam NUM_FEAS_MI = 5;
    localparam DIR_DW = 2;
    localparam EMB_DW = 2;
    localparam QRS_EMB_LEN = 24;
    localparam T_EMB_LEN = 29;
    localparam FEATURE_DIM = 22;
    localparam FEATURE_DIM_MI = 64;    
    localparam ACTIVATION_BUF_LEN1 = (DCNN1_LENGTH_OUT-2)*DATA_BQ_DW; 
    localparam ACTIVATION_BUF_LEN2 = INPUT_DW*LENGTH_IN;
    localparam ACTIVATION_BUF_LEN3 = NUM_FEAS_MI * (NUM_LEADS+1) * FEATURE_SUM_DW + DIR_DW * NUM_LEADS + NUM_LEADS * (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW;
    localparam ACTIVATION_BUF_LEN4 = FEATURE_DIM_MI*FEATURE_SUM_DW;            



    // wire [$clog2(PE_NUM+1)-1:0] cnt_cho_32;
    // wire encoder_out_vld; //encoder
    // wire  signed [2*DATA_OUT_DW-1: 0] lstm_hidden_cat; //lstm out
    wire signed [1:0] softmax_out;
    // wire signed [DATA_OUT_DW-1: 0] decoder_out; // dcnn1, cnn11, cnn12
    // wire [2*DATA_OUT_DW-1: 0] decoder_out_cat; //dcnn2, cnn21, cnn22
    // wire lstm_hidden_unit_vld;
    wire decoder_out_vld;
    wire dcnn1_temp_value_vld;
    wire dcnn1_transfer_temp_value_en;
    wire dcnn1_temp_rst;
    wire [1:0] encoder_shift_en;
    // wire lstm_xt_shift_en;
    // wire [2*PE_NUM-1:0] shift_crl_all;
    // wire [PE_NUM-1:0] cnt_bt_all;
    // wire [DATA_OUT_DW*ACTIVATION_BUF_LEN1-1:0] act_sr1;
    // wire [DATA_OUT_DW*ACTIVATION_BUF_LEN2-1:0] act_sr2;
    // wire [DATA_OUT_DW*ACTIVATION_BUF_LEN3-1:0] act_sr3;
    // wire [DATA_OUT_DW*ACTIVATION_BUF_LEN4-1:0] act_sr4;
    wire [ACTIVATION_BUF_LEN1-1:0] act_sr1;
    wire [ACTIVATION_BUF_LEN2-1:0] act_sr2;
    wire [ACTIVATION_BUF_LEN3-1:0] act_sr3;
    wire [ACTIVATION_BUF_LEN4-1:0] act_sr4;
    wire [5*(INIT_NUM_BEATS+1)*INPUT_DW+2*(INIT_NUM_BEATS+1) * INTEVAL_DW-1:0] feature_rb;

    // post
    localparam NUM_WAVE = 4;
    // localparam LABEL_DW = 2;
    localparam TREND_DW = 4;
    localparam BG_MIN_LEN = 15;
    localparam PQRST_MIN_LEN = 3;
    
    wire [$clog2(LENGTH_IN+1)-1:0] wave_duration; // post out glb in
    wire modify_en; // post out glb in
    wire connection_shift; // post out glb in
    wire refine_shift_re;
    wire refine_shift;
    wire emb_shift;

    wire [$clog2(LENGTH_IN+1)-1:0] r_loc;
    wire signed [INPUT_DW-1:0] r_amp;
    wire [$clog2(LENGTH_IN+1)-1:0] t_on_loc;
    wire signed [INPUT_DW-1:0] t_on_amp;
    wire [$clog2(LENGTH_IN+1)-1:0] t_off_loc;
    wire signed [INPUT_DW-1:0] t_off_amp;
    wire [$clog2(LENGTH_IN+1)-1:0] t_loc;
    wire signed [INPUT_DW-1:0] t_amp;
    wire signed [DIR_DW-1:0] t_dir;
    wire [$clog2(LENGTH_IN+1)-1:0] p_on_loc;
    wire signed [INPUT_DW-1:0] p_on_amp;
    wire [$clog2(LENGTH_IN+1)-1:0] p_off_loc;
    wire signed [INPUT_DW-1:0] p_off_amp;
    wire [$clog2(LENGTH_IN+1)-1:0] p_loc;
    wire signed [INPUT_DW-1:0] p_amp;
    wire signed [DIR_DW-1:0] p_dir;
    wire [$clog2(LENGTH_IN+1)-1:0] q_loc;
    wire signed [INPUT_DW-1:0] q_amp;
    wire [$clog2(LENGTH_IN+1)-1:0] pq_loc;
    wire signed [INPUT_DW-1:0] pq_amp;
    wire [$clog2(LENGTH_IN+1)-1:0] s_loc;
    wire signed [INPUT_DW-1:0] s_amp;
    wire [$clog2(LENGTH_IN+1)-1:0] st_loc;
    wire signed [INPUT_DW-1:0] st_amp;
    wire signed [INPUT_DW-1:0] st_amp_1;
    wire signed [INPUT_DW-1:0] st_amp_2;
    wire signed [INPUT_DW-1:0] st_amp_4;
    wire signed [INPUT_DW-1:0] st_amp_6;
    wire signed [INPUT_DW-1:0] iso_line;
    wire  [EMB_DW*QRS_EMB_LEN-1:0] qrs_emb_buffer;
    wire  [EMB_DW*T_EMB_LEN-1:0] t_emb_buffer;
    wire [3:0] post_state;
    wire [4:0] refine_state;
    // assign post_rdy = network_rdy|(((feature_done) & (cnt_lead!= NUM_LEADS-1))|((top_state_c == idle) & (cnt_lead== NUM_LEADS-1))); //need to change
    // assign post_rdy = decoder_done|(((feature_done) & (cnt_lead!= NUM_LEADS-1))|((top_state_c == idle) & (cnt_lead== NUM_LEADS-1)));
    assign post_rdy = decoder_done|(((input_finish) & (cnt_lead!= 0))); //change
    //  assign post_rdy = input_finish;
    // feature
    
      


    assign  feature_rdy = post_done;

    
    wire [1:0] save_fea_en;
    wire  signed [INTEVAL_DW-1:0] rr_diff;
    wire [INTEVAL_DW-1:0] qrs;
    wire signed [INTEVAL_DW -1 : 0] rr_pre_rr_ave;
    wire signed [INTEVAL_DW -1 : 0] rr_post_rr_ave;
    wire signed [INTEVAL_DW -1 : 0] qrs_cur_qrs_ave;
    wire signed [INPUT_DW - 1: 0] r_amp_r_amp_ave;
    wire signed [INPUT_DW - 1: 0] q_amp_q_amp_ave;
    wire signed [INPUT_DW - 1: 0] s_amp_s_amp_ave;
    wire signed [INPUT_DW - 1: 0] p_amp_p_amp_ave;
    wire signed [INPUT_DW - 1: 0] t_amp_t_amp_ave;

    reg [INTEVAL_DW-1:0] rr_pre_d;
    reg signed [INPUT_DW-1:0] r_amp_d;
    reg signed [INPUT_DW-1:0] t_amp_d;
    reg signed [INPUT_DW-1:0] p_amp_d;
    reg signed [INPUT_DW-1:0] q_amp_d;
    reg [INTEVAL_DW-1:0] qrs_d;
    reg signed [INPUT_DW-1:0] s_amp_d;

    // wire signed [INPUT_DW - 1: 0] r_amp_t_amp;
    wire signed [INPUT_DW - 1: 0] q_amp_iso;
    wire signed [INPUT_DW - 1: 0] s_amp_iso;
    wire signed [INPUT_DW - 1: 0] t_amp_iso;
    // wire signed [INPUT_DW - 1: 0] r_amp_iso;
    wire signed [INPUT_DW - 1: 0] st_amp_iso;
    wire signed [INPUT_DW - 1: 0] st_slo;
    // wire signed [FEATURE_SUM_DW - 1: 0] r_amp_t_amp_sum;
    wire signed [FEATURE_SUM_DW - 1: 0] q_amp_iso_sum;
    wire signed [FEATURE_SUM_DW - 1: 0] s_amp_iso_sum;
    wire signed [FEATURE_SUM_DW - 1: 0] t_amp_iso_sum;
    // wire signed [FEATURE_SUM_DW - 1: 0] r_amp_iso_sum;
    wire signed [FEATURE_SUM_DW - 1: 0] st_amp_iso_sum;
    wire signed [FEATURE_SUM_DW - 1: 0] st_slo_sum;
    reg [INTEVAL_DW-1:0] rr_pre;
    

    always @(posedge wclk or negedge rst_n) begin
        if(!rst_n) begin
            rr_pre_d <= 0;
            qrs_d <= 0;
            r_amp_d <= 0;
            t_amp_d <= 0;
            p_amp_d <= 0;
            q_amp_d <= 0;
            s_amp_d <= 0;
        end
        else begin
            if (top_state_c == done) begin
                rr_pre_d <= rr_pre;
                qrs_d <= qrs;
                r_amp_d <= r_amp;
                t_amp_d <= t_amp;
                p_amp_d <= p_amp;
                q_amp_d <=q_amp;
                s_amp_d <= s_amp;
            end  
            else;          
        end
    end

    
    localparam  ANN_HIDDEN_DIM= 32;
    localparam  ANN_OUT_DIM = 5;
    localparam  ANN_OUT_DIM_MI = 2;
    
    wire ann_out_vld;
    wire ann_hidden_out_vld;
    wire ann_relu_en;

    wire [FEATURE_DIM*INPUT_DW-1:0] feature_matrix;
    wire [2*NUM_FEAS_MI * FEATURE_SUM_DW + (QRS_EMB_LEN + T_EMB_LEN) * EMB_DW + DIR_DW -1:0] feature_matrix_mi;  
    wire [1:0] ann_shift;
    wire feature_shift;
    wire input_init_en;
    wire ann_mi_1;
    wire ann_mi_2;
 
    wire [4 : 0] ann_state;
    wire signed [SRAM16_DW +INPUT_DW  -1:0] ann_out;
    wire signed [SRAM16_DW + FEATURE_SUM_DW -1:0] ann_out_mi;
    wire signed [SRAM16_DW +INPUT_DW  -1:0] ann_hidden_in;
    wire signed [FEATURE_SUM_DW  -1:0] ann_mi_in;
    wire signed [FEATURE_SUM_DW +SRAM16_DW  -1:0] ann_mi_hidden_in;

    // assign  ann_rdy = (mode == 0)?(init_features_end & feature_done) : ((cnt_lead_d == NUM_LEADS-1) & feature_done); // NEED to change
    assign ann_rdy = (mode == 0)?(init_features_end & feature_done) :((cnt_lead_d == NUM_LEADS-1) & feature_done); // NEED to change

    reg [ARR_LABEL_DW-1:0] predict_pre;
    always @(posedge wclk  or negedge rst_n) begin
        if (!rst_n) predict_pre <= 0;
        else begin
            if (ann_done) begin
                predict_pre <= arr_type;
            end
            else begin
                predict_pre <= predict_pre;
            end

        end
    end
/////////////////////////// SPI ////////////////////////////
   wire spi_dout_vld;
   wire [SPI_DW-1:0] spi_dout;
   wire [SPI_DW-1:0] spi_din;
   reg [PARAM_DW-1:0]  LEAD_THRES;
   
   reg [INTEVAL_DW-1:0] rr_post;
   reg [INPUT_DW*LENGTH_IN-1:0] input_signal;
//    wire signed [INPUT_DW -1:0] input_signal_mem[LENGTH_IN-1:0];
//    genvar   test_id;
//    generate
//    for (test_id = 0; test_id < LENGTH_IN; test_id = test_id + 1) begin: test
//        assign input_signal_mem[test_id] = input_signal[test_id*(INPUT_DW)+(INPUT_DW)-1- :(INPUT_DW)];
//    end        
//    endgenerate
    wire [1:0] encoder_shift_en_sck;
   assign encoder_shift_en_sck = encoder_shift_en & {wclk, wclk};

   always @(posedge sck or negedge rst_n) begin
       if (!rst_n) input_signal <= 0;
       else begin
           if (top_state_c == data_init) begin
                if (spi_dout_vld & (spi_dout[SPI_DW-1:SPI_DW-3] == 3'b000)) 
                    input_signal <= {spi_dout[INPUT_DW-1:0], input_signal[INPUT_DW*LENGTH_IN-1:INPUT_DW]};
                else input_signal <= input_signal;
           end
           else if (top_state_c == seg_network) begin
               
                if (seg_state == 4'b0001) begin
                    if (encoder_shift_en_sck == 1) 
                        input_signal <= {input_signal[ENCODER_STRIDE*INPUT_DW-1:0], input_signal[LENGTH_IN* INPUT_DW -1:ENCODER_STRIDE* INPUT_DW]}; // shift for input
                    else if (encoder_shift_en_sck == 2)
                        input_signal <= {input_signal[INPUT_DW-1:0], input_signal[LENGTH_IN* INPUT_DW -1: INPUT_DW]}; // shift for input
                    else if (encoder_shift_en_sck == 3)
                        input_signal <= {input_signal[7*INPUT_DW-1:0], input_signal[LENGTH_IN* INPUT_DW -1:7* INPUT_DW]}; // shift for input
                    else
                        input_signal <= input_signal ;
                     
                end
           end
           else;
       end
   end
   always @(posedge sck or negedge rst_n) begin
       if (!rst_n) begin
           rr_pre <= 0;
           rr_post <= 0;
       end
       else begin
           if (spi_dout_vld & (top_state_c == data_init)) begin
                if  (spi_dout[SPI_DW-1:SPI_DW-3] == 3'b001) rr_pre <= spi_dout[INTEVAL_DW-1:0];
                else if (spi_dout[SPI_DW-1:SPI_DW-3] == 3'b010) rr_post <=  spi_dout[INTEVAL_DW-1:0];
                else;
           end
           else; 
       end
   end
   always @(posedge sck or negedge rst_n) begin
       if (!rst_n) begin
           LEAD_THRES <= 0;
       end
       else begin
           if (spi_dout_vld & (top_state_c == data_init)) begin
                if  (spi_dout[SPI_DW-1:SPI_DW-3] == 3'b011) LEAD_THRES <= spi_dout[PARAM_DW-1:0];
                else;
           end
           else; 
       end
   end

   wire softmax_vld;
   localparam SOFTMAX_REG_DW = 5;
   assign softmax_vld = decoder_out_vld & !cnn22_is_first_2d & (decoder_top_state ==  3'b010) & wclk;
   reg [2:0] cnt_softmax;
   reg [SPI_DW - SOFTMAX_REG_DW*2 -1:0] addr_softmax;
   reg [SOFTMAX_REG_DW*2 -1:0] softmax_spi_reg;
   always @(posedge sck or negedge rst_n) begin
       if (!rst_n) begin
           cnt_softmax <= 0;
           softmax_spi_reg <= 0;
       end
       else begin
            if (softmax_vld) begin
                cnt_softmax <= (cnt_softmax == SOFTMAX_REG_DW -1)? 0:  cnt_softmax + 1;
                softmax_spi_reg <= {softmax_out, softmax_spi_reg[SOFTMAX_REG_DW*2 -1:2]};
            end
            else if (one_beat_done) begin //rst
                cnt_softmax <= 0;
                softmax_spi_reg <= 0;                
            end
            else begin
                cnt_softmax <= cnt_softmax;
                softmax_spi_reg <= softmax_spi_reg;
            end
       end
   end

   always @(posedge sck or negedge rst_n) begin
       if (!rst_n) begin
           softmax_rdy <= 0;
       end
       else begin
            if ((cnt_softmax  == SOFTMAX_REG_DW -1) & (softmax_vld) ) begin
                softmax_rdy <= 1;
            end
            else begin
                softmax_rdy <= 0;
            end
       end
   end

   always @(posedge sck or negedge rst_n) begin
       if (!rst_n) begin
           addr_softmax <= 0;
       end
       else begin
            if (softmax_rdy) begin
                addr_softmax <=  addr_softmax + 1;
            end                
            else if (one_beat_done)  begin
                addr_softmax <= 0;
            end
            else begin
                addr_softmax <= addr_softmax;
            end
       end
   end   
   assign spi_din = {addr_softmax , softmax_spi_reg};

   SPI_SLAVE #(
   .SPI_DW(SPI_DW)
   )spi_slave_u(
   .wclk(sck),
   .rst_n(rst_n),
   .spi_clk(spi_clk),
   .cs_n(spi_cs_n),
   .mosi(spi_mosi),
   .spi_din(spi_din), // not used
   .dout_vld(spi_dout_vld),
   .miso(spi_miso), // not used
   .spi_dout(spi_dout)
   );

/////////////////////////// SPI ///////////////////////////


    SEG_NETWORK #(
    .ENCODER_WB_DW (ENCODER_WB_DW),
    .SRAM16_DW(SRAM16_DW),
    .SRAM32_DW(SRAM32_DW),
    .SRAM8_DW(SRAM8_DW),
    .SRAM8192_AW(SRAM8192_AW),
    .SRAM1024_AW(SRAM1024_AW),
    .SRAM512_AW(SRAM512_AW),
    .INPUT_DW(INPUT_DW),
    .DATA_OUT_DW(DATA_OUT_DW),
    .DATA_BQ_DW(DATA_BQ_DW),//new
    .LSTM_WU_DW(LSTM_WU_DW), 
    .LSTM_B_DW(LSTM_B_DW), 
    .DECODER_W_DW(DECODER_W_DW),
    .DECODER_SCALE_DW(DECODER_SCALE_DW),
    .DECODER_B_DW(DECODER_B_DW),
    .SPAD_DEPTH(SPAD_DEPTH),
    .LSTM_SCALE_DW (LSTM_SCALE_DW),
    .ENCODER_SCALE_DW(ENCODER_SCALE_DW),
    .PE_NUM(PE_NUM), //new
    .ENCODER_LENGTH_IN(LENGTH_IN),
    
    .ENCODER_STRIDE(ENCODER_STRIDE),
    .DCNN1_LENGTH_OUT(DCNN1_LENGTH_OUT),
    .CNN22_LENGTH_OUT(CNN22_LENGTH_OUT)
    )
    SEG_NETWORK_u(
    // system
    .sclk(sclk),
    .wclk(wclk),
    .rst_n(rst_n),
    
    .network_rdy(network_rdy), ///need to change
    // .encoder_out(encoder_out),
    //input/output(global) buffer
    .input_signal(input_signal),
    
    .seg_state(seg_state),
    
    
    // sram
    .sram1_dout(sram1_dout),
    .sram2_dout(sram2_dout),
    .sram3_dout(sram3_dout),
    .sram4_dout(sram4_dout),
    .sram5_dout(sram5_dout),
    .sram6_dout(sram6_dout),
    .addr_sram(addr_sram_seg),
    .sram1_en(seg_sram1_en), // 1 is enabled
    .sram2_en(seg_sram2_en),
    .sram3_en(seg_sram3_en), // 1 is enabled
    .sram4_en(seg_sram4_en),
    .sram5_en(seg_sram5_en),
    .sram6_en(seg_sram6_en),
    .lstm_top_state(lstm_top_state),
    .decoder_top_state(decoder_top_state),

    .addr_encoder_w_init(addr_encoder_w_init), // from top.v
    .addr_encoder_b_init(addr_encoder_b_init), // from top.v
    .addr_encoder_output_scale(addr_encoder_output_scale), // from top.v
    .addr_lstm_w00_init(addr_lstm_w00_init),
    .addr_lstm_u00_init(addr_lstm_u00_init),
    .addr_lstm_b00_init(addr_lstm_b00_init),
    .addr_lstm_scales_00_init( addr_lstm_scales_00_init),
    .addr_lstm_w01_init(addr_lstm_w01_init),
    .addr_lstm_u01_init(addr_lstm_u01_init),
    .addr_lstm_b01_init(addr_lstm_b01_init),
    .addr_lstm_scales_01_init( addr_lstm_scales_01_init),
    .addr_lstm_w10_init(addr_lstm_w10_init),
    .addr_lstm_u10_init(addr_lstm_u10_init),
    .addr_lstm_b10_init(addr_lstm_b10_init),
    .addr_lstm_scales_10_init( addr_lstm_scales_10_init),        
    .addr_lstm_w11_init(addr_lstm_w11_init),
    .addr_lstm_u11_init(addr_lstm_u11_init), 
    .addr_lstm_b11_init(addr_lstm_b11_init), 
    .addr_lstm_scales_11_init( addr_lstm_scales_11_init),    

    .addr_dcnn1_scales_init(addr_dcnn1_scales_init),
    .addr_dcnn1_w_init(addr_dcnn1_w_init),    
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
    
    // communication with SPAD_W
    .spad1_w_we_en(spad1_w_we_en_seg), 
    .spad_w_we_en_2_32(spad_w_we_en_2_32),

    .spad_w_addr_re(spad_w_addr_re_seg),
    .spad_w_addr_we(spad_w_addr_we_seg),
    
    //communication with SPRD_A
    .spad_a_addr_re(spad_a_addr_re_seg),
    .spad1_a_data_in (spad1_a_data_in_seg), //new
    .spad17_a_data_in (spad17_a_data_in), //new
    .spad18_a_data_in (spad18_a_data_in), //new
    .spad19_a_data_in (spad19_a_data_in), //new
    .spad20_a_data_in (spad20_a_data_in), //new
    .spad21_a_data_in (spad21_a_data_in), //new
    .spad22_a_data_in (spad22_a_data_in), //new
    .spad23_a_data_in (spad23_a_data_in), //new
    .spad24_a_data_in (spad24_a_data_in), //new
    .spad25_a_data_in (spad25_a_data_in), //new
    .spad26_a_data_in (spad26_a_data_in), //new
    .spad27_a_data_in (spad27_a_data_in), //new
    .spad28_a_data_in (spad28_a_data_in), //new
    .spad29_a_data_in (spad29_a_data_in), //new
    .spad30_a_data_in (spad30_a_data_in), //new
    .spad31_a_data_in (spad31_a_data_in), //new
    .spad32_a_data_in (spad32_a_data_in), //new
    // .spad_a_addr_we_all(spad_a_addr_we_all),
    .spad_a_addr_we(spad_a_addr_we),
    .spad_a_we_en_all(spad_a_we_en_all),
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

    .encoder_b(encoder_b),
    .encoder_w(encoder_w),
    .lstm_wu(lstm_wu), //new
    .encoder_out(encoder_out),
    .dcnn1_temp_value_for_1(dcnn1_temp_value_for_1),
    .decoder_w(decoder_w),
    // .decoder_dcnn1_b(decoder_dcnn1_b),
    // .decoder_scale(decoder_scale),
    .decoder_b1(decoder_b1),
    .decoder_b2(decoder_b2),

    .pe_out_32b_all(psum_32b_all),//need to change order
    .pe_out_a(pe_out_a),//new
    .pe_out_b(pe_out_b),//new
    .mult_a_out_round(mult_a_out_round),
    .mult_b_out_round(mult_b_out_round),
    .pe_out_sum_a_final(pe_out_sum_a_final),//new
    .pe_out_sum_b_final(pe_out_sum_b_final),//new
    // .out_temp_A_final(out_temp_A_final),
    .hardmard_a_all(hardmard_a_all),//new
    .hardmard_b_all(hardmard_b_all),//new
    .lstm_b(lstm_b),//new
    .out_bq(out_bq),//new
    .scale(scale),//new
    .out_bq2(out_bq2),//new
    .scale2(scale2),//new
    .lstm_ct_temp_out_cat(lstm_ct_temp_out_cat),
    .encoder_scale(encoder_scale),//new

    .cnn22_is_first(cnn22_is_first),
    .cnn22_is_first_2d(cnn22_is_first_2d),
    // .seg_out(seg_out),
    .mult_a_crl(mult_a_crl_seg),//new
    .mult_b_crl(mult_b_crl),//new
    .add_a_crl(add_a_crl),//new
    .add_b_crl(add_b_crl),//new
    .mult_int8_crl_1_16(mult_int8_crl_1_16),//new
    .mult_int8_crl_17_32(mult_int8_crl_17_32),
    .decoder_mult_int8_crl(decoder_mult_int8_crl),
    .mult_out_round_en(mult_out_round_en),
    .pe_out_sum_a_final_en(pe_out_sum_a_final_en),
    .pe_out_sum_b_final_en(pe_out_sum_b_final_en),
    .encoder_relu_en(encoder_relu_en),
    .encoder_round_en(encoder_round_en),
    // .seg_out_vld(seg_out_vld),
    .decoder_done(decoder_done),
    
    // .cnt_cho_32(cnt_cho_32),
    // .encoder_out_vld(encoder_out_vld),
    // .lstm_hidden_cat(lstm_hidden_cat),
    .softmax_out(softmax_out),
    // .decoder_out(decoder_out),
    // .decoder_out_cat(decoder_out_cat),
    // .lstm_hidden_unit_vld(lstm_hidden_unit_vld),
    .decoder_out_vld(decoder_out_vld),
    .dcnn1_temp_value_vld(dcnn1_temp_value_vld),
    .dcnn1_transfer_temp_value_en(dcnn1_transfer_temp_value_en),
    .dcnn1_temp_rst(dcnn1_temp_rst),
    .encoder_shift_en(encoder_shift_en),
    // .lstm_xt_shift_en(lstm_xt_shift_en),
    // .shift_crl_all(shift_crl_all),
    // .cnt_bt_all(cnt_bt_all),
    .act_sr1_1(act_sr1[DATA_BQ_DW-1:0]),
    .sram_act_dout(sram_act_dout),
    .sram_act_din(sram_act_din),
    .sram_act_en(sram_act_en),
    .sram_act_we(sram_act_we),
    .addr_sram_act(addr_sram_act)
    // .act_sr2(act_sr2),
    // .act_sr3(act_sr3),
    // .act_sr4(act_sr4)
    // .addr_sram_act_test(addr_sram_act_test),
    // .sram_act_din_test(sram_act_din_test),
    // .sram_act_we_test(sram_act_we_test)
    
    );
    GLB_BUF_S #(
    .INPUT_DW(INPUT_DW),
    .DATA_OUT_DW(DATA_OUT_DW),
    .DATA_BQ_DW(DATA_BQ_DW),
    .FEATURE_SUM_DW (INPUT_DW + 4),
    .SPAD_DEPTH(SPAD_DEPTH),
    .INTEVAL_DW($clog2(LENGTH_IN+1)),
    .SAVE_NUM_BEATS (INIT_NUM_BEATS + 1),
    .NUM_FEAS_MI(NUM_FEAS_MI),
    .NUM_LEADS(NUM_LEADS),
    // .CNN21_OUT(CNN21_OUT),
    .ENCODER_LENGTH_IN(LENGTH_IN),
    .CNN22_LENGTH_OUT(CNN22_LENGTH_OUT),
    .DIR_DW(DIR_DW),
    .LABEL_DW(LABEL_DW),
    .EMB_DW(EMB_DW),
    .QRS_EMB_LEN(QRS_EMB_LEN),
    .T_EMB_LEN(T_EMB_LEN),
    .ANN_WB_DW(SRAM16_DW),
    .FEATURE_DIM(FEATURE_DIM),
    .FEATURE_DIM_MI(FEATURE_DIM_MI),
    .ANN_HIDDEN_DIM(ANN_HIDDEN_DIM),
    .ACTIVATION_BUF_LEN1(ACTIVATION_BUF_LEN1),
    .ACTIVATION_BUF_LEN2(ACTIVATION_BUF_LEN2),
    .ACTIVATION_BUF_LEN4(ACTIVATION_BUF_LEN4)

    )
    GLB_BUF_S_u(
    .wclk(wclk),
    .rst_n(rst_n),
    .input_signal(input_signal),
    // .softmax_out_all(softmax_out_all),
    // .cnn21_out(cnn21_out),
    .top_state(top_state_c),
    .seg_state(seg_state),
    // .lstm_top_state(lstm_top_state),
    .decoder_top_state(decoder_top_state),
    .ann_state(ann_state),
    .cnn22_is_first_2d(cnn22_is_first_2d),
    .pe_out_32b_1(psum_32b_all[DATA_BQ_DW-1:0]),
    .softmax_out(softmax_out),
    .decoder_out_vld(decoder_out_vld),
    .dcnn1_temp_value_vld(dcnn1_temp_value_vld),
    .dcnn1_transfer_temp_value_en(dcnn1_transfer_temp_value_en),
    .dcnn1_temp_rst(dcnn1_temp_rst),
    .act_sr1(act_sr1),
    .act_sr2(act_sr2),
    .act_sr3(act_sr3),
    .act_sr4(act_sr4),
    .feature_rb(feature_rb),
    .post_state(post_state),
    .refine_state(refine_state),
    .wave_duration(wave_duration),
    .modify_en(modify_en),
    .connection_shift(connection_shift),
    .refine_shift_re(refine_shift_re), // post out  glb in
    .refine_shift(refine_shift), // post out  glb in    
    .emb_shift(emb_shift),
    .feature_shift(feature_shift),
    .save_fea_en(save_fea_en),
    .rr_pre_d(rr_pre_d),
    .r_amp_d(r_amp_d),
    .t_amp_d(t_amp_d),
    .p_amp_d(p_amp_d),
    .q_amp_d(q_amp_d),
    .qrs_d(qrs_d),
    .s_amp_d(s_amp_d),
    .rr_pre(rr_pre),
    .r_amp(r_amp),
    .t_amp(t_amp),
    .p_amp(p_amp),
    .q_amp(q_amp),
    .qrs(qrs),
    .s_amp(s_amp),
    .q_amp_iso(q_amp_iso),
    .s_amp_iso(s_amp_iso),
    .t_amp_iso(t_amp_iso),
    .st_amp_iso(st_amp_iso),
    .st_slo(st_slo),
    .q_amp_iso_sum(q_amp_iso_sum),
    .s_amp_iso_sum(s_amp_iso_sum),
    .t_amp_iso_sum(t_amp_iso_sum),
    .st_amp_iso_sum(st_amp_iso_sum),
    .st_slo_sum(st_slo_sum),
    .qrs_emb_buffer(qrs_emb_buffer),
    .t_emb_buffer(t_emb_buffer),
    .t_dir(t_dir),
    .cnt_lead(cnt_lead),
    .ann_shift(ann_shift),
    .input_init_en(input_init_en),
    .ann_mi_1(ann_mi_1),
    .ann_mi_2(ann_mi_2),
    .feature_matrix(feature_matrix),
    .feature_matrix_mi(feature_matrix_mi),
    .ann_out_vld(ann_out_vld),
    .ann_hidden_out_vld(ann_hidden_out_vld),
    .ann_out(ann_out),
    .ann_out_mi(ann_out_mi)
    );
    
    
    HL55LPHDSP8192x8B1M16W1SA10	 SRAM_8192x8_act (
    .Q    (sram_act_dout) ,
    .CLK   (sclk) ,
    .ME  (sram_act_en) ,
    .WE  (sram_act_we) ,
    .TEST1  (1'b0) , // disable
    .RME  (1'b0) ,// disable
    .ADR  (addr_sram_act) ,
    .WEM(8'B1111_1111),
    .D    (sram_act_din),
    .RM(4'B0000));

    HL55LPHDSP1024x32B1M4W1SA10	 SRAM_1024x32_u1 (
    .Q    (sram1_dout) ,
    .CLK   (sclk) ,
    .ME  (sram1_en) ,
    .WE  (sram1_we) ,
    .TEST1  (1'b0) , // disable
    .RME  (1'b0) ,// disable
    .ADR  (addr_sram[9:0]) ,
    .WEM(32'B1111_1111_1111_1111_1111_1111_1111_1111),
    .D    (sram1_din),
    .RM(4'B0000));

    HL55LPHDSP8192x16B1M16W1SA10	 SRAM_8192x16_u2 (
    .Q    (sram2_dout) ,
    .CLK   (sclk) ,
    .ME  (sram2_en) ,
    .WE  (sram2_we) ,
    .TEST1  (1'b0) , // disable
    .RME  (1'b0) ,// disable
    .ADR  (addr_sram) ,
    .WEM(16'B1111_1111_1111_1111),
    .D    (sram2_din),
    .RM(4'B0000));


    HL55LPHDSP8192x16B1M16W1SA10	 SRAM_8192x16_u3 (
    .Q    (sram3_dout) ,
    .CLK   (sclk) ,
    .ME  (sram3_en) ,
    .WE  (sram3_we) ,
    .TEST1  (1'b0) , // disable
    .RME  (1'b0) ,// disable
    .ADR  (addr_sram) ,
    .WEM(16'B1111_1111_1111_1111),
    .D    (sram3_din),
    .RM(4'B0000));

    HL55LPHDSP8192x16B1M16W1SA10	 SRAM_8192x16_u4 (
    .Q    (sram4_dout) ,
    .CLK   (sclk) ,
    .ME  (sram4_en) ,
    .WE  (sram4_we) ,
    .TEST1  (1'b0) , // disable
    .RME  (1'b0) ,// disable
    .ADR  (addr_sram) ,
    .WEM(16'B1111_1111_1111_1111),
    .D    (sram4_din),
    .RM(4'B0000));

    HL55LPHDSP8192x16B1M16W1SA10	 SRAM_8192x16_u5 (
    .Q    (sram5_dout) ,
    .CLK   (sclk) ,
    .ME  (sram5_en) ,
    .WE  (sram5_we) ,
    .TEST1  (1'b0) , // disable
    .RME  (1'b0) ,// disable
    .ADR  (addr_sram) ,
    .WEM(16'B1111_1111_1111_1111),
    .D    (sram5_din),
    .RM(4'B0000));

    HL55LPHDSP512x16B1M4W1SA10	 SRAM_512x16_u6 (
    .Q    (sram6_dout) ,
    .CLK   (sclk) ,
    .ME  (sram6_en) ,
    .WE  (sram6_we) ,
    .TEST1  (1'b0) , // disable
    .RME  (1'b0) ,// disable
    .ADR  (addr_sram[8:0]) ,
    .WEM(16'B1111_1111_1111_1111),
    .D    (sram6_din),
    .RM(4'B0000));

    HL55LPHDSP1024x16B1M4W1SA10	 SRAM_1024x16_u7 (
    .Q    (sram7_dout) ,
    .CLK   (sclk) ,
    .ME  (sram7_en) ,
    .WE  (sram7_we) ,
    .TEST1  (1'b0) , // disable
    .RME  (1'b0) ,// disable
    .ADR  (addr_sram[9:0]) ,
    .WEM(16'B1111_1111_1111_1111),
    .D    (sram7_din),
    .RM(4'B0000));

    HL55LPHDSP8192x16B1M16W1SA10	 SRAM_8192x16_u8 (
    .Q    (sram8_dout) ,
    .CLK   (sclk) ,
    .ME  (sram8_en) ,
    .WE  (sram8_we) ,
    .TEST1  (1'b0) , // disable
    .RME  (1'b0) ,// disable
    .ADR  (addr_sram) ,
    .WEM(16'B1111_1111_1111_1111),
    .D    (sram8_din),
    .RM(4'B0000));

    HL55LPHDSP8192x16B1M16W1SA10	 SRAM_8192x16_u9 (
    .Q    (sram9_dout) ,
    .CLK   (sclk) ,
    .ME  (sram9_en) ,
    .WE  (sram9_we) ,
    .TEST1  (1'b0) , // disable
    .RME  (1'b0) ,// disable
    .ADR  (addr_sram) ,
    .WEM(16'B1111_1111_1111_1111),
    .D    (sram9_din),
    .RM(4'B0000));

    HL55LPHDSP8192x16B1M16W1SA10	 SRAM_8192x16_u10 (
    .Q    (sram10_dout) ,
    .CLK   (sclk) ,
    .ME  (sram10_en) ,
    .WE  (sram10_we) ,
    .TEST1  (1'b0) , // disable
    .RME  (1'b0) ,// disable
    .ADR  (addr_sram) ,
    .WEM(16'B1111_1111_1111_1111),
    .D    (sram10_din),
    .RM(4'B0000));

    HL55LPHDSP1024x16B1M4W1SA10	 SRAM_1024x16_u11 (
    .Q    (sram11_dout) ,
    .CLK   (sclk) ,
    .ME  (sram11_en) ,
    .WE  (sram11_we) ,
    .TEST1  (1'b0) , // disable
    .RME  (1'b0) ,// disable
    .ADR  (addr_sram[9:0]) ,
    .WEM(16'B1111_1111_1111_1111),
    .D    (sram11_din),
    .RM(4'B0000));

    HL55LPHDSP512x16B1M4W1SA10 SRAM_512x16_u12(
    .Q    (sram12_dout) ,
    .CLK   (sclk) ,
    .ME  (sram12_en) ,
    .WE  (sram12_we) ,
    .TEST1  (1'b0) , // disable
    .RME  (1'b0) ,// disable
    .ADR  (addr_sram[8:0]) ,
    .WEM(16'B1111_1111_1111_1111),
    .D    (sram12_din),
    .RM(4'B0000));


  

    PE_MAIN #(
    .WB_DW (ENCODER_WB_DW),
    .A_DW (INPUT_DW),
    .ENCODER_SCALE_DW (ENCODER_SCALE_DW),
    .LSTM_SCALE_DW(LSTM_SCALE_DW),
    .DATA_BQ_DW(DATA_BQ_DW),
    .ANN_WB_DW(SRAM16_DW),
    .OUT_DW(DATA_OUT_DW),
    .FEATURE_SUM_DW(FEATURE_SUM_DW)
    )
    PE_MAIN_u(//system
    .wclk(wclk),
    .rst_n(rst_n),
    .spad_w_data(spad1_w_data_out), //encoder
    .spad_a_data(spad1_a_data_out), //encoder
    .encoder_b(encoder_b), //encoder
    .encoder_scale(encoder_scale), 
    .lstm_b(lstm_b),
    .lstm_ct_temp_in_cat(lstm_ct_temp_out_cat),//gates_scale * f_t[hs] * c_t[hs], from ct_buffer
    .out_bq(out_bq), // lstm: from each PEs
    .scale(scale),
    .out_bq2(out_bq2), // lstm: from each PEs
    .scale2(scale2),
    .dcnn1_temp_value_for_1(dcnn1_temp_value_for_1),
    .psum_32b_8(psum_out_32b_8),
    .psum_32b_16(psum_out_32b_16),
    .psum_32b_24(psum_out_32b_24_d),
    .psum_32b_32(psum_out_32b_32),
    .psum_32b_32_d(psum_out_32b_32_d),
    .seg_state(seg_state), //0000:idle, 0001:encoder, 0010:lstm
    .mult_a_crl(mult_a_crl),  // encoder: 00: idle, 01: mac, 11: add_b, 10: requantization; lstm->00:idle, 01:requantization gates, 11 requan tail  
    .mult_b_crl(mult_b_crl), // encoder:  00: idle, 01: mac, 11: add_b, 10: requantization; lstm->00:idle, 01:requantization gates, 11 requan tail
    .mult_int8_crl(mult_int8_crl_all[2:0]), //lstm-> 000:idle/reset, 001:gatess_mac, 011: transfer, 111:hold, 010: hardmard prod
    .add_a_crl(add_a_crl), //lstm-> 00:idle,  01: add gates, 11: add_tail; decoder: 00idle 10 ,add_b
    .add_b_crl(add_b_crl), //lstm-> 00:idle,  01: add gates, 11: add_tail
    .cnn22_is_first(cnn22_is_first),
    // .decoder_dcnn1_b(decoder_dcnn1_b),
    // .decoder_scale(decoder_scale),
    .decoder_b1(decoder_b1),
    .decoder_b2(decoder_b2),
    .decoder_top_state(decoder_top_state),

    .hardmard_a(hardmard_a_all[DATA_OUT_DW-1: 0]),
    .hardmard_b(hardmard_b_all[DATA_OUT_DW-1: 0]),
    .encoder_out(encoder_out),
    .sum_a_final(pe_out_sum_a_final),
    .sum_b_final(pe_out_sum_b_final),
    // .out_temp_A_final(out_temp_A_final),
    .lstm_hardmard_temp_a(pe_out_a),
    .lstm_hardmard_temp_b(pe_out_b),
    .mult_a_out_round(mult_a_out_round),
    .mult_b_out_round(mult_b_out_round),
    .psum_32b(psum_32b_all[DATA_BQ_DW-1:0]),
    .out_32b(pe_out_32b_all[DATA_BQ_DW-1:0]),
    .mult_out_round_en(mult_out_round_en),
    .encoder_relu_en(encoder_relu_en),
    .encoder_round_en(encoder_round_en),
    .sum_a_final_en(pe_out_sum_a_final_en),
    .sum_b_final_en(pe_out_sum_b_final_en),
    .top_state(top_state_c),
    .ann_state(ann_state),
    .ann_b(ann_b),
    .ann_mi_1(ann_mi_1),
    .ann_mi_2(ann_mi_2),
    .ann_hidden_in(ann_hidden_in),
    .ann_mi_hidden_in(ann_mi_hidden_in),
    .ann_mi_in(ann_mi_in),
    .ann_out(ann_out),
    .ann_out_mi(ann_out_mi),
    .ann_relu_en(ann_relu_en)
    );

    genvar p;
    generate
        for (p = 1; p < PE_NUM; p = p+1) begin: gen_pe_1_16
            PE_NORM #(
            .DATA_DW(DATA_OUT_DW),
            .OUT_BQ_DW(DATA_BQ_DW)
            )
            PE_NORM_u(
            .wclk(wclk),
            .rst_n(rst_n),
            .seg_state(seg_state),
            .spad_w_data(spad_w_data_out_all[p*DATA_OUT_DW-1-:DATA_OUT_DW]),
            .spad_a_data(spad_a_data_out_all[p*DATA_OUT_DW-1-:DATA_OUT_DW]),
            .hardmard_a(hardmard_a_all[(p+1)*DATA_OUT_DW-1-:DATA_OUT_DW]),//hardmard multiplier
            .hardmard_b(hardmard_b_all[(p+1)*DATA_OUT_DW-1-:DATA_OUT_DW]),//hardmard multiplier
            .mult_int8_crl(mult_int8_crl_all[(p+1)*3-1-:3]), // lstm-> 000:idle/reset, 001:gatess_mac, 011: transfer, 111:hold, 010: hardmard prod
            .psum_32b(psum_32b_all[(p+1)*DATA_BQ_DW-1-:DATA_BQ_DW]), // from adjacent pe
            .out_32b(pe_out_32b_all[(p+1)*DATA_BQ_DW-1-:DATA_BQ_DW]));        
        end
    endgenerate




    SPAD_A #(
    .DATA_DW(INPUT_DW),
    .DEPTH(SPAD_DEPTH)
    )
    SPAD_A_MAIN_u(
    .sclk(sclk),
    .rst_n(rst_n),
    .is_sram_in(is_sram_in_all[0]),               //低电平有效的复位信号
    .sram_data_in(spad1_a_data_sram_in),
    .data_in(spad1_a_data_in),    //
    .we_en(spad_a_we_en_all[0]),
    // .addr_we(spad_a_addr_we_all[$clog2(SPAD_DEPTH)-1:0]),
    .addr_we(spad_a_addr_we),
    .addr_re(spad_a_addr_re_all[$clog2(SPAD_DEPTH)-1:0]),//
    .data_out(spad1_a_data_out));//
    genvar a;
    generate
        for (a = 0; a < PE_NUM-1; a = a+1) begin: gen_spad_a_module
            SPAD_A #(
            .DATA_DW(DATA_OUT_DW),
            .DEPTH(SPAD_DEPTH)
            )
            SPAD_A_u(
            .sclk(sclk),
            .rst_n(rst_n),  
            .is_sram_in(is_sram_in_all[a+1]),              //低电平有效的复位信号
            .sram_data_in(spad_a_data_sram_in_all[(a+1)*DATA_OUT_DW-1-:DATA_OUT_DW]),
            .data_in(spad_a_data_in_all[(a+1)*DATA_OUT_DW*SPAD_DEPTH-1-:DATA_OUT_DW*SPAD_DEPTH]),    //
            .we_en(spad_a_we_en_all[a+1]),
            // .addr_we(spad_a_addr_we_all[(a+2)*($clog2(SPAD_DEPTH))-1-:($clog2(SPAD_DEPTH))]),
            .addr_we(spad_a_addr_we),
            .addr_re(spad_a_addr_re_all[(a+2)*($clog2(SPAD_DEPTH))-1-:($clog2(SPAD_DEPTH))]),//
            .data_out(spad_a_data_out_all[(a+1)*DATA_OUT_DW-1-:DATA_OUT_DW]));//        
        end
    endgenerate

    SPAD_W #(
    .WEIGHT_DW(ENCODER_WB_DW),
    .DEPTH(SPAD_DEPTH))
    
    SPAD_W_MAIN_u(
    .sclk(sclk),
    .rst_n(rst_n),       //reset, 0 is effective
    .data_in(spad1_w_data_in),     //                       //读使能信号，高电平有�??
    .we_en(spad1_w_we_en),  // 1 is write,0 is read      //                                                     //写使能信号，高电平有�??
    .addr_we(spad_w_addr_we), //
    .addr_re(spad_w_addr_re_all[$clog2(SPAD_DEPTH)-1:0]), //
    .data_out(spad1_w_data_out));//
    
    genvar w;
    generate
        for (w = 0; w <PE_NUM-1; w = w + 1) begin: gen_spad_w_module
            SPAD_W #(
            .WEIGHT_DW(LSTM_WU_DW),
            .DEPTH(SPAD_DEPTH))
            
            SPAD_W_u(
            .sclk(sclk),
            .rst_n(rst_n),       //reset, 0 is effective
            .data_in(spad_w_data_in),     //                       //读使能信号，高电平有�??
            .we_en(spad_w_we_en_2_32[w]),  // 1 is write,0 is read      //                                                     //写使能信号，高电平有�??
            .addr_we(spad_w_addr_we), //
            .addr_re(spad_w_addr_re_all[(w+2)*$clog2(SPAD_DEPTH)-1-:$clog2(SPAD_DEPTH)]), //
            .data_out(spad_w_data_out_all[(w+1)*DATA_OUT_DW-1-:DATA_OUT_DW]));//        
        end
    endgenerate
    

    POST #(
    .INPUT_DW(INPUT_DW),
    .DATA_DW(DATA_OUT_DW),
    .LENGTH_IN(LENGTH_IN),
    .INTEVAL_DW(INTEVAL_DW),
    .NUM_WAVE(NUM_WAVE),
    .LABEL_DW(LABEL_DW),
    .TREND_DW(TREND_DW),
    .DIR_DW(DIR_DW),
    .EMB_DW(EMB_DW),
    .QRS_EMB_LEN(QRS_EMB_LEN),
    .T_EMB_LEN(T_EMB_LEN),
    .BG_MIN_LEN(BG_MIN_LEN),
    .PQRST_MIN_LEN(PQRST_MIN_LEN),
    .ACTIVATION_BUF_LEN1(ACTIVATION_BUF_LEN1),
    .ACTIVATION_BUF_LEN2(ACTIVATION_BUF_LEN2)
    )POST_u(
    .wclk(wclk),
    .rst_n(rst_n),
    .post_rdy(post_rdy),
    .act_sr1(act_sr1),
    .act_sr2(act_sr2),
    .post_state(post_state),
    .refine_state(refine_state),
    .wave_duration(wave_duration),
    .modify_en(modify_en), // post out  glb in
    .connection_shift(connection_shift),// post out  glb in
    .refine_shift_re(refine_shift_re), // post out  glb in
    .refine_shift(refine_shift), // post out  glb in
    .emb_shift(emb_shift),
    .r_loc(r_loc), 
    .r_amp_final(r_amp),
    .t_on_loc(t_on_loc),
    .t_on_amp(t_on_amp),
    .t_off_loc(t_off_loc),
    .t_off_amp(t_off_amp),
    .t_loc(t_loc),
    .t_amp_final(t_amp),
    .t_dir(t_dir),
    .p_on_loc(p_on_loc),
    .p_on_amp(p_on_amp),
    .p_off_loc(p_off_loc),
    .p_off_amp(p_off_amp),
    .p_loc(p_loc),
    .p_amp_final(p_amp),
    .p_dir(p_dir),
    .q_loc(q_loc),
    .q_amp_final(q_amp),
    .pq_loc(pq_loc),
    .pq_amp(pq_amp),
    .s_loc(s_loc),
    .s_amp_final(s_amp),
    .st_loc(st_loc),
    .st_amp(st_amp),
    .post_done(post_done),
    .mode(mode),
    .feature_done(feature_done),
    .ann_done(ann_done),
    .st_amp_1(st_amp_1),
    .st_amp_2(st_amp_2),
    .st_amp_4(st_amp_4),
    .st_amp_6(st_amp_6),
    .iso_line(iso_line),
    .qrs_emb_buffer(qrs_emb_buffer),
    .t_emb_buffer(t_emb_buffer)
    );

    FEATURE # (
    .INPUT_DW(INPUT_DW),
    .DATA_DW(DATA_OUT_DW),
    .LENGTH_IN(LENGTH_IN),
    .ARR_LABEL_DW(ARR_LABEL_DW),
    .INIT_NUM_BEATS(INIT_NUM_BEATS),
    .NUM_FEAS_MI(NUM_FEAS_MI),
    .INTEVAL_DW(INTEVAL_DW),
    // .ACTIVATION_BUF_LEN1(ACTIVATION_BUF_LEN1),
    .NUM_LEADS(NUM_LEADS),
    .FEATURE_SUM_DW (INPUT_DW + 4)
    )    

    feature_u (
    .wclk(wclk),
    .rst_n(rst_n),
    .feature_rdy(feature_rdy),
    .feature_rb(feature_rb),
    .rr_pre(rr_pre),
    .rr_post(rr_post),
    .r_amp(r_amp),
    .t_amp(t_amp),
    .p_amp(p_amp),
    .q_amp(q_amp),
    .q_loc(q_loc),
    .s_loc(s_loc),
    .s_amp(s_amp),
    .predict_pre(predict_pre), // 0  norm
    .save_fea_en(save_fea_en),
    .rr_diff(rr_diff),
    .qrs(qrs),
    .rr_pre_rr_ave(rr_pre_rr_ave),
    .rr_post_rr_ave(rr_post_rr_ave),
    .qrs_cur_qrs_ave(qrs_cur_qrs_ave),
    .r_amp_r_amp_ave(r_amp_r_amp_ave),
    .q_amp_q_amp_ave(q_amp_q_amp_ave),
    .s_amp_s_amp_ave(s_amp_s_amp_ave),
    .p_amp_p_amp_ave(p_amp_p_amp_ave),
    .t_amp_t_amp_ave(t_amp_t_amp_ave),
    .feature_done(feature_done),

    .init_features_end(init_features_end),
    .mode(mode),
    .cnt_lead(cnt_lead),
    .r_loc(r_loc),
    .t_loc(t_loc),
    .st_loc(st_loc),
    .st_amp(st_amp),
    .st_amp_1(st_amp_1),
    .st_amp_2(st_amp_2),
    .st_amp_4(st_amp_4),
    .st_amp_6(st_amp_6),
    .iso_line(iso_line),

    // .r_amp_t_amp(r_amp_t_amp),
    .q_amp_iso(q_amp_iso),
    .s_amp_iso(s_amp_iso),
    .t_amp_iso(t_amp_iso),
    // .r_amp_iso(r_amp_iso),
    .st_amp_iso(st_amp_iso),
    .st_slo(st_slo),
    // .r_amp_t_amp_sum(r_amp_t_amp_sum),
    .q_amp_iso_sum(q_amp_iso_sum),
    .s_amp_iso_sum(s_amp_iso_sum),
    .t_amp_iso_sum(t_amp_iso_sum),
    // .r_amp_iso_sum(r_amp_iso_sum),
    .st_amp_iso_sum(st_amp_iso_sum),
    .st_slo_sum(st_slo_sum)
    );

    ANN #( 
    .INPUT_DW(INPUT_DW),
    .DATA_DW(DATA_OUT_DW),
    .FEATURE_SUM_DW(FEATURE_SUM_DW),
    .LENGTH_IN(LENGTH_IN),
    .SRAM1024_AW(SRAM1024_AW),
    .SRAM8192_AW(SRAM8192_AW),
    .SRAM512_AW(SRAM512_AW),
    .SRAM16_DW(SRAM16_DW),
    .ARR_LABEL_DW(ARR_LABEL_DW),
    .DIR_DW(DIR_DW),
    .SPAD_DEPTH(SPAD_DEPTH),
    .INTEVAL_DW(INTEVAL_DW),
    .NUM_FEAS_MI(NUM_FEAS_MI),
    .FEATURE_DIM(FEATURE_DIM),
    .FEATURE_DIM_MI(FEATURE_DIM_MI),
    .ANN_HIDDEN_DIM(ANN_HIDDEN_DIM),
    .ANN_OUT_DIM(ANN_OUT_DIM),
    .ANN_OUT_DIM_MI(ANN_OUT_DIM_MI),
    .NUM_LEADS(NUM_LEADS),
    .EMB_DW(EMB_DW),
    .QRS_EMB_LEN(QRS_EMB_LEN),
    .T_EMB_LEN(T_EMB_LEN),
    .PARAM_DW(PARAM_DW),
    .ACTIVATION_BUF_LEN1(ACTIVATION_BUF_LEN1),
    .ACTIVATION_BUF_LEN3(ACTIVATION_BUF_LEN3),
    .ACTIVATION_BUF_LEN4(ACTIVATION_BUF_LEN4)
    ) ann_u (
    .wclk(wclk),
    .sclk(sclk),
    .rst_n(rst_n),
    .ann_rdy(ann_rdy), 
    .act_sr1(act_sr1),
    .act_sr3(act_sr3),
    .act_sr4(act_sr4),
    .ann_state(ann_state),
    .addr_ann1_w_init(addr_ann1_w_init),
    .addr_ann1_b_init(addr_ann1_b_init),
    .addr_ann2_w_init(addr_ann2_w_init),
    .addr_ann2_b_init(addr_ann2_b_init),
    .addr_ann1_1_w_init(addr_ann1_1_w_init),
    .addr_ann1_1_b_init(addr_ann1_1_b_init),
    .addr_ann2_1_w_init(addr_ann2_1_w_init),
    .addr_ann2_1_b_init(addr_ann2_1_b_init),
    .addr_ann3_1_w_init(addr_ann3_1_w_init),
    .addr_ann3_1_b_init(addr_ann3_1_b_init),
    .addr_ann4_1_w_init(addr_ann4_1_w_init),
    .addr_ann4_1_b_init(addr_ann4_1_b_init),
    .addr_ann5_1_w_init(addr_ann5_1_w_init),
    .addr_ann5_1_b_init(addr_ann5_1_b_init),
    .addr_ann6_1_w_init(addr_ann6_1_w_init),
    .addr_ann6_1_b_init(addr_ann6_1_b_init),
    .addr_ann7_1_w_init(addr_ann7_1_w_init),
    .addr_ann7_1_b_init(addr_ann7_1_b_init),
    .addr_ann8_1_w_init(addr_ann8_1_w_init),
    .addr_ann8_1_b_init(addr_ann8_1_b_init),
    .addr_ann9_1_w_init(addr_ann9_1_w_init),
    .addr_ann9_1_b_init(addr_ann9_1_b_init),
    .addr_ann10_1_w_init(addr_ann10_1_w_init),
    .addr_ann10_1_b_init(addr_ann10_1_b_init),
    .addr_ann11_1_w_init(addr_ann11_1_w_init),
    .addr_ann11_1_b_init(addr_ann11_1_b_init),
    .addr_ann12_1_w_init(addr_ann12_1_w_init),
    .addr_ann12_1_b_init(addr_ann12_1_b_init),
    .addr_ann1_2_w_init(addr_ann1_2_w_init),
    .addr_ann1_2_b_init(addr_ann1_2_b_init),
    .addr_ann2_2_w_init(addr_ann2_2_w_init),
    .addr_ann2_2_b_init(addr_ann2_2_b_init),
    .addr_ann3_2_w_init(addr_ann3_2_w_init),
    .addr_ann3_2_b_init(addr_ann3_2_b_init),
    .addr_ann4_2_w_init(addr_ann4_2_w_init),
    .addr_ann4_2_b_init(addr_ann4_2_b_init),
    .addr_ann5_2_w_init(addr_ann5_2_w_init),
    .addr_ann5_2_b_init(addr_ann5_2_b_init),
    .addr_ann6_2_w_init(addr_ann6_2_w_init),
    .addr_ann6_2_b_init(addr_ann6_2_b_init),
    .addr_ann7_2_w_init(addr_ann7_2_w_init),
    .addr_ann7_2_b_init(addr_ann7_2_b_init),
    .addr_ann8_2_w_init(addr_ann8_2_w_init),
    .addr_ann8_2_b_init(addr_ann8_2_b_init),
    .addr_ann9_2_w_init(addr_ann9_2_w_init),
    .addr_ann9_2_b_init(addr_ann9_2_b_init),
    .addr_ann10_2_w_init(addr_ann10_2_w_init),
    .addr_ann10_2_b_init(addr_ann10_2_b_init),
    .addr_ann11_2_w_init(addr_ann11_2_w_init),
    .addr_ann11_2_b_init(addr_ann11_2_b_init),
    .addr_ann12_2_w_init(addr_ann12_2_w_init),
    .addr_ann12_2_b_init(addr_ann12_2_b_init),
    .sram7_dout(sram7_dout),
    .sram8_dout(sram8_dout),
    .sram9_dout(sram9_dout),
    .sram10_dout(sram10_dout),
    .sram11_dout(sram11_dout),
    .sram12_dout(sram12_dout),
    .sram7_en(ann_sram7_en),
    .sram8_en(ann_sram8_en),
    .sram9_en(ann_sram9_en),
    .sram10_en(ann_sram10_en),
    .sram11_en(ann_sram11_en),
    .sram12_en(ann_sram12_en),
    .LEAD_THRES(LEAD_THRES),
    .addr_ann(addr_sram_ann),
    .ann_w(ann_w),
    .ann_b(ann_b),
    .ann_shift(ann_shift),
    .init_features_end(init_features_end),
    .input_init_en(input_init_en),
    .spad_w_we_en(spad1_w_we_en_ann), 
    .spad_w_addr_re(spad_w_addr_re_ann), 
    .spad_w_addr_we(spad_w_addr_we_ann), 
    .spad_a_addr_re(spad_a_addr_re_ann), 
    .spad_a_data_in(spad1_a_data_in_ann), 
    .mult_a_crl(mult_a_crl_ann),
    .ann_out_vld(ann_out_vld),
    .ann_hidden_out_vld(ann_hidden_out_vld),
    .rr_diff(rr_diff),
    .rr_pre_rr_ave(rr_pre_rr_ave),
    .rr_post_rr_ave(rr_post_rr_ave),
    .qrs_cur_qrs_ave(qrs_cur_qrs_ave),
    .r_amp_r_amp_ave(r_amp_r_amp_ave),
    .q_amp_q_amp_ave(q_amp_q_amp_ave),
    .s_amp_s_amp_ave(s_amp_s_amp_ave),
    .p_amp_p_amp_ave(p_amp_p_amp_ave),
    .t_amp_t_amp_ave(t_amp_t_amp_ave),
    .t_dir(t_dir),
    .feature_matrix(feature_matrix) ,
    .feature_shift(feature_shift),
    .feature_matrix_mi(feature_matrix_mi),
    .ann_hidden_in(ann_hidden_in),
    .ann_relu_en(ann_relu_en),
    .ann_done(ann_done),
    .arr_type(arr_type),
    .ann_out(ann_out),
    .mode(mode),
    .ann_mi_1(ann_mi_1),
    .ann_mi_2(ann_mi_2),

    .ann_mi_in(ann_mi_in),
    .ann_mi_hidden_in(ann_mi_hidden_in),
    .mi_type(mi_type),
    .ann_out_mi(ann_out_mi)
    );
endmodule
