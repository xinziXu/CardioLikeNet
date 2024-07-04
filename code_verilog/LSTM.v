module LSTM #( parameter DATA_DW = 8,
    INPUT_DW = 12,
    DATA_BQ_DW = 32,
    WU_DW = 8,
    B_DW = 32,
    SCALE_DW = 32,
    NUM_LAYERS = 2,
    NUM_DIRECTIONS = 2,
    HS = 32,
    INPUT_SIZE = 32,
    SRAM1024_AW = 10,
    SRAM8192_AW = 13,
    SRAM8_DW = 8,
    SRAM32_DW = 32, 
    SRAM16_DW = 16,
    SPAD_DEPTH = 8,  
    PE_NUM = 32, 
    T = 64,
    ADDR_ENCODER_SRAM_ACT_INIT = 0,
    ADDR_LSTM10_SRAM_ACT_INIT = T * INPUT_SIZE + 2 * T * HS,
    ADDR_LSTM11_SRAM_ACT_INIT = 0  )
    // ACTIVATION_BUF_LEN1 = HS * T,
    // ACTIVATION_BUF_LEN2 = INPUT_SIZE * T,
    // ACTIVATION_BUF_LEN3 = HS * T ) 
    (input wclk,
    input sclk,
    input rst_n,
    // input [ACTIVATION_BUF_LEN2-1:0] act_sr2, //from segment.v, XT_0, HT_0_REVERSE
    // input [ACTIVATION_BUF_LEN1-1:0] act_sr1, // from segment.v , HT_0
    // input [ACTIVATION_BUF_LEN3*DATA_DW-1:0] act_sr3, // from segment.v , HT_0
    input [SRAM8192_AW-1:0] addr_lstm_w00_init,    // from top.v, w0
    input [SRAM8192_AW-1:0] addr_lstm_u00_init,    // from top.v, w0
    input [SRAM1024_AW-1:0] addr_lstm_b00_init,    // from top.v, b0
    input [SRAM1024_AW-1:0] addr_lstm_scales_00_init,  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale
    input [SRAM8192_AW-1:0] addr_lstm_w01_init,    // from top.v, w0
    input [SRAM8192_AW-1:0] addr_lstm_u01_init,    // from top.v, w0
    input [SRAM1024_AW-1:0] addr_lstm_b01_init,    // from top.v, b0
    input [SRAM1024_AW-1:0] addr_lstm_scales_01_init,  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale
    input [SRAM8192_AW-1:0] addr_lstm_w10_init,    // from top.v, w0
    input [SRAM8192_AW-1:0] addr_lstm_u10_init,    // from top.v, w0
    input [SRAM1024_AW-1:0] addr_lstm_b10_init,    // from top.v, b0
    input [SRAM1024_AW-1:0] addr_lstm_scales_10_init,  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale
    input [SRAM8192_AW-1:0] addr_lstm_w11_init,    // from top.v, w0
    input [SRAM8192_AW-1:0] addr_lstm_u11_init,    // from top.v, w0
    input [SRAM1024_AW-1:0] addr_lstm_b11_init,    // from top.v, b0
    input [SRAM1024_AW-1:0] addr_lstm_scales_11_init,  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale

    input [SRAM32_DW-1 : 0] sram1_dout,   //sram1_dout: b and scales
    input [SRAM16_DW-1 : 0] sram2_dout,   //sram2_dout:  w and u
    input [SRAM16_DW-1 : 0] sram3_dout,   //sram2_dout:  w and u
    input [SRAM16_DW-1 : 0] sram4_dout,   //sram2_dout:  w and u
    input lstm_rdy, // segment.v

    // output signed [2*DATA_DW-1: 0] lstm_hidden_cat, //ht
    // output lstm_hidden_unit_vld,
    output [SRAM8192_AW-1:0] addr_sram,    // data width of weight and bias are the same, so no need to differenciate
    output sram1_en,
    output sram2_en,
    output sram3_en,
    output sram4_en,
    output reg [3-1 : 0] lstm_top_state,

    output signed [WU_DW-1 : 0] lstm_wu, //segment.v
    output lstm_done,         // lstm completed
    // output xt_shift_en, //segment.v
    output spad1_w_we_en, // the enable signal for the first spad
    output [PE_NUM-2:0] spad_w_we_en_2_32, //the enable signal for the rest of PEs
    output [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_re, //paralell, the same
    output [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_we,  // serial operation, so share one addr_we
    output [$clog2(SPAD_DEPTH)-1 : 0] spad_a_addr_re, //paralell, the same
    // output [$clog2(SPAD_DEPTH)*PE_NUM/2-1 : 0] spad_a_addr_we_1_16, //new
    output [$clog2(SPAD_DEPTH)-1 : 0] spad_a_addr_we, //new
    output [PE_NUM/2-1 : 0] spad_a_we_en_1_16, //new
    output [INPUT_DW -1 : 0] spad1_a_data_sram_in,
    output [DATA_DW -1 : 0] spad2_a_data_sram_in, 
    output [DATA_DW -1 : 0] spad3_a_data_sram_in,
    output [DATA_DW -1 : 0] spad4_a_data_sram_in,
    output [DATA_DW -1 : 0] spad5_a_data_sram_in,
    output [DATA_DW -1 : 0] spad6_a_data_sram_in,
    output [DATA_DW -1 : 0] spad7_a_data_sram_in,
    output [DATA_DW -1 : 0] spad8_a_data_sram_in,
    output [DATA_DW -1 : 0] spad9_a_data_sram_in,
    output [DATA_DW -1 : 0] spad10_a_data_sram_in,
    output [DATA_DW -1 : 0] spad11_a_data_sram_in,
    output [DATA_DW -1 : 0] spad12_a_data_sram_in,
    output [DATA_DW -1 : 0] spad13_a_data_sram_in,
    output [DATA_DW -1 : 0] spad14_a_data_sram_in,
    output [DATA_DW -1 : 0] spad15_a_data_sram_in,
    output [DATA_DW -1 : 0] spad16_a_data_sram_in,
    output [DATA_DW*SPAD_DEPTH -1 : 0] spad17_a_data_in,
    output [DATA_DW*SPAD_DEPTH -1 : 0] spad18_a_data_in,
    output [DATA_DW*SPAD_DEPTH -1 : 0] spad19_a_data_in,
    output [DATA_DW*SPAD_DEPTH -1 : 0] spad20_a_data_in,
    output [DATA_DW*SPAD_DEPTH -1 : 0] spad21_a_data_in,
    output [DATA_DW*SPAD_DEPTH -1 : 0] spad22_a_data_in,
    output [DATA_DW*SPAD_DEPTH -1 : 0] spad23_a_data_in,
    output [DATA_DW*SPAD_DEPTH -1 : 0] spad24_a_data_in,
    output [DATA_DW*SPAD_DEPTH -1 : 0] spad25_a_data_in,
    output [DATA_DW*SPAD_DEPTH -1 : 0] spad26_a_data_in,
    output [DATA_DW*SPAD_DEPTH -1 : 0] spad27_a_data_in,
    output [DATA_DW*SPAD_DEPTH -1 : 0] spad28_a_data_in,
    output [DATA_DW*SPAD_DEPTH -1 : 0] spad29_a_data_in,
    output [DATA_DW*SPAD_DEPTH -1 : 0] spad30_a_data_in,  
    output [DATA_DW*SPAD_DEPTH -1 : 0] spad31_a_data_in,
    output [DATA_DW*SPAD_DEPTH -1 : 0] spad32_a_data_in,
    input [PE_NUM*DATA_BQ_DW-1: 0] pe_out_32b_all,
    input signed [2*DATA_DW+SCALE_DW-1: 0] pe_out_a,
    input signed [2*DATA_DW+SCALE_DW-1: 0] pe_out_b,
    input signed [DATA_DW-1: 0] mult_a_out_round,
    input signed [DATA_DW-1: 0] mult_b_out_round,
    input signed [DATA_DW-1:0] pe_out_sum_a_final,
    input signed [DATA_DW-1:0] pe_out_sum_b_final,    
    output signed [PE_NUM * DATA_DW-1: 0] hardmard_a_all,
    output signed [PE_NUM * DATA_DW-1: 0] hardmard_b_all,
    output signed [B_DW-1: 0] lstm_b,
    output signed [DATA_BQ_DW-1:0] out_bq, // lstm: from each PEs
    output signed  [SCALE_DW -1 : 0] scale,
    output signed [DATA_BQ_DW-1:0] out_bq2, // lstm: from each PEs
    output signed [SCALE_DW -1 : 0] scale2,
    output signed [2*(2*DATA_DW+SCALE_DW)-1: 0] lstm_ct_temp_out_cat,
    
    output [1:0] mult_a_crl, // 00:idle, 01:requantize     
    output [1:0] mult_b_crl,
    output [1:0] add_a_crl,
    output [1:0] add_b_crl,
    output [2:0] mult_int8_crl_1_16,// lstm-> 000:idle/reset, 001:gatess_mac, 011: transfer, 111:hold, 010: hardmard prod
    output [2:0] mult_int8_crl_17_32,
    output mult_out_round_en,
    output pe_out_sum_a_final_en,
    output pe_out_sum_b_final_en,
    input  [SRAM8_DW-1:0]  sram_act_dout,
    output [SRAM8192_AW -1 : 0] addr_sram_act,
    output sram_act_en,
    output sram_act_we,
    output [SRAM8_DW-1:0] sram_act_din);
    
    localparam N_state = NUM_LAYERS * NUM_DIRECTIONS + 2;
    localparam N = $clog2(N_state+1);
    localparam idle = 3'd0;
    localparam layer_00 = 3'd1;
    localparam layer_01 = 3'd2;
    localparam layer_10 = 3'd3;
    localparam layer_11 = 3'd4;
    localparam done = 3'd5;

    
    reg [N-1 : 0] lstm_top_state_next;     
    wire layer_done;
    assign lstm_done = (lstm_top_state == done);

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            lstm_top_state <= idle;
        else
            lstm_top_state <= lstm_top_state_next;
    end

    always @(*) begin
        case(lstm_top_state)
        idle: begin
            if (lstm_rdy) lstm_top_state_next = layer_00; // need to be changed
            else lstm_top_state_next = idle;
        end
        layer_00: begin
            if (layer_done) lstm_top_state_next = layer_01;
            else lstm_top_state_next = layer_00;            
        end
        layer_01:begin
            if (layer_done) lstm_top_state_next = layer_10;
            else lstm_top_state_next = layer_01;              
        end
        layer_10: begin
            if (layer_done) lstm_top_state_next = layer_11;
            else lstm_top_state_next = layer_10;              
        end
        layer_11: begin
            if (layer_done) lstm_top_state_next = done;
            else lstm_top_state_next = layer_11;             
        end
        done: lstm_top_state_next = idle; 
        default:lstm_top_state_next = idle; 
        endcase
    end
    reg layer_done_d;
    reg lstm_rdy_d;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            layer_done_d <= 0;
            lstm_rdy_d <= 0;
            end
        else begin
            layer_done_d <= layer_done;
            lstm_rdy_d <= lstm_rdy;
        end
           
    end    
    wire layer_rdy;
    assign layer_rdy = lstm_rdy_d | (layer_done_d & (lstm_top_state != done)); // layer_done_d is because need to back to idle
    
    


    reg [SRAM8192_AW-1:0] addr_lstm_w_init;    // from top.v, w0
    reg [SRAM8192_AW-1:0] addr_lstm_u_init;    // from top.v, w0
    reg [SRAM1024_AW-1:0] addr_lstm_b_init;    // from top.v, b0
    reg [SRAM1024_AW-1:0] addr_lstm_scales_init;  // from top.v , SwSx_Sg, 
    always @(*) begin
        case(lstm_top_state_next)
        idle: begin
            addr_lstm_w_init = addr_lstm_w00_init;//change
            addr_lstm_u_init = addr_lstm_u00_init;
            addr_lstm_b_init = addr_lstm_b00_init;
            addr_lstm_scales_init  =   addr_lstm_scales_00_init; // need to correspond with the first state
        end
        layer_00: begin
            addr_lstm_w_init = addr_lstm_w00_init;
            addr_lstm_u_init = addr_lstm_u00_init;
            addr_lstm_b_init = addr_lstm_b00_init;
            addr_lstm_scales_init  =   addr_lstm_scales_00_init;           
        end
        layer_01:begin
            addr_lstm_w_init = addr_lstm_w01_init;
            addr_lstm_u_init = addr_lstm_u01_init;
            addr_lstm_b_init = addr_lstm_b01_init;
            addr_lstm_scales_init  =   addr_lstm_scales_01_init;           
        end
        layer_10: begin
            addr_lstm_w_init = addr_lstm_w10_init;
            addr_lstm_u_init = addr_lstm_u10_init;
            addr_lstm_b_init = addr_lstm_b10_init;
            addr_lstm_scales_init  =   addr_lstm_scales_10_init;                
        end
        layer_11: begin
            addr_lstm_w_init = addr_lstm_w11_init;
            addr_lstm_u_init = addr_lstm_u11_init;
            addr_lstm_b_init = addr_lstm_b11_init;
            addr_lstm_scales_init  =   addr_lstm_scales_11_init;            
        end
        default: begin
            addr_lstm_w_init = addr_lstm_w00_init;//change
            addr_lstm_u_init = addr_lstm_u00_init;
            addr_lstm_b_init = addr_lstm_b00_init;
            addr_lstm_scales_init  =   addr_lstm_scales_00_init; // need to correspond with the first state            
        end
        endcase
    end
    LSTM_UNIT #(
    .DATA_DW (DATA_DW),
    .INPUT_DW(INPUT_DW),
    .DATA_BQ_DW(DATA_BQ_DW),
    .WU_DW (WU_DW),
    .B_DW(B_DW),
    .SCALE_DW (SCALE_DW),
    .HS (HS),
    .INPUT_SIZE (INPUT_SIZE),
    .SRAM1024_AW (SRAM1024_AW),
    .SRAM8192_AW (SRAM8192_AW),
    .SRAM32_DW (SRAM32_DW),
    .SRAM8_DW(SRAM8_DW),
    .SRAM16_DW (SRAM16_DW),
    .SPAD_DEPTH (SPAD_DEPTH),  
    .PE_NUM (PE_NUM), 
    .T (T),
    .ADDR_ENCODER_SRAM_ACT_INIT(ADDR_ENCODER_SRAM_ACT_INIT)
    // .ACTIVATION_BUF_LEN1( ACTIVATION_BUF_LEN1),
    // .ACTIVATION_BUF_LEN2(ACTIVATION_BUF_LEN2 ),
    // .ACTIVATION_BUF_LEN3(ACTIVATION_BUF_LEN3 )
    )
    LSTM_layer(
    .wclk(wclk),
    .sclk(sclk),
    .rst_n(rst_n),
    .lstm_top_state(lstm_top_state),
    // .act_sr2(act_sr2),
    // .act_sr1(act_sr1),
    // .act_sr3(act_sr3),
    .addr_lstm_w_init(addr_lstm_w_init),    // from top.v, w0, u0
    .addr_lstm_u_init(addr_lstm_u_init),    // from top.v, w0, u0
    .addr_lstm_b_init(addr_lstm_b_init),    // from top.v, b0, b0_reverse, b1, b1_reverse
    .addr_lstm_scales_init(addr_lstm_scales_init),  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale

    .sram1_dout(sram1_dout),
    .sram2_dout(sram2_dout),
    .sram3_dout(sram3_dout),
    .sram4_dout(sram4_dout),

    .layer_rdy(layer_rdy), //top


    // .lstm_hidden_cat(lstm_hidden_cat), //ht
    // .lstm_hidden_unit_vld(lstm_hidden_unit_vld),
    .addr_sram(addr_sram),    // data width of weight and bias are the same, so no need to differenciate
    .sram1_en(sram1_en),
    .sram2_en(sram2_en),
    .sram3_en(sram3_en),
    .sram4_en(sram4_en),

    .lstm_wu(lstm_wu), //segment
    .layer_done(layer_done),         // layer completed
    // .xt_shift_en( xt_shift_en), //segment
    .spad1_w_we_en( spad1_w_we_en), // the enable signal for the first spad
    .spad_w_we_en_2_32( spad_w_we_en_2_32), //the enable signal for the rest of PEs
    .spad_w_addr_re( spad_w_addr_re), //paralell, the same
    .spad_w_addr_we( spad_w_addr_we),  // serial operation, so share one addr_we
    .spad_a_addr_re( spad_a_addr_re), //paralell, the same
    // .spad_a_addr_we_1_16(spad_a_addr_we_1_16),
    .spad_a_addr_we(spad_a_addr_we),
    .spad_a_we_en_1_16(spad_a_we_en_1_16),
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


    .spad17_a_data_in( spad17_a_data_in),
    .spad18_a_data_in( spad18_a_data_in), 
    .spad19_a_data_in( spad19_a_data_in),
    .spad20_a_data_in( spad20_a_data_in),  
    .spad21_a_data_in( spad21_a_data_in),
    .spad22_a_data_in( spad22_a_data_in), 
    .spad23_a_data_in( spad23_a_data_in),
    .spad24_a_data_in( spad24_a_data_in), 
    .spad25_a_data_in( spad25_a_data_in),
    .spad26_a_data_in( spad26_a_data_in), 
    .spad27_a_data_in( spad27_a_data_in),
    .spad28_a_data_in( spad28_a_data_in), 
    .spad29_a_data_in( spad29_a_data_in),
    .spad30_a_data_in( spad30_a_data_in),
    .spad31_a_data_in( spad31_a_data_in),
    .spad32_a_data_in( spad32_a_data_in),     
    .pe_out_32b_all(pe_out_32b_all),
    .pe_out_a(pe_out_a),
    .pe_out_b(pe_out_b),
    .mult_a_out_round(mult_a_out_round),
    .mult_b_out_round(mult_b_out_round),    
    .pe_out_sum_a_final(pe_out_sum_a_final),
    .pe_out_sum_b_final(pe_out_sum_b_final),    
    .hardmard_a_all(hardmard_a_all),
    .hardmard_b_all(hardmard_b_all),
    .lstm_b( lstm_b),
    .out_bq( out_bq), // lstm: from each PEs
    .scale( scale),
    .out_bq2( out_bq2), // lstm: from each PEs
    .scale2( scale2),
    .lstm_ct_temp_out_cat(lstm_ct_temp_out_cat),

    .mult_a_crl( mult_a_crl), // 00:idle, 01:requantize     
    .mult_b_crl( mult_b_crl),
    .add_a_crl( add_a_crl),
    .add_b_crl( add_b_crl),     
    .mult_int8_crl_1_16( mult_int8_crl_1_16),
    .mult_int8_crl_17_32( mult_int8_crl_17_32),
    .mult_out_round_en(mult_out_round_en),
    .pe_out_sum_a_final_en(pe_out_sum_a_final_en),
    .pe_out_sum_b_final_en(pe_out_sum_b_final_en),
    .sram_act_dout(sram_act_dout),
    .sram_act_din(sram_act_din),
    .sram_act_en(sram_act_en),
    .sram_act_we(sram_act_we),
    .addr_sram_act(addr_sram_act)    );

endmodule