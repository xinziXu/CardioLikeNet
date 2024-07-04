`timescale  1ns/100ps
module DECODER_L #(parameter DATA_DW = 8,
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
    CNN12_LENGTH_OUT = DCNN1_LENGTH_OUT,
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

    // input [ACTIVATION_BUF_LEN2*DATA_DW-1:0] act_sr2, //from segment.v, XT_0, HT_0_REVERSE
    input [DATA_BQ_DW-1:0] act_sr1_1, // from segment.v , HT_0
    // input [ACTIVATION_BUF_LEN3*DATA_DW-1:0] act_sr3,
    // input [ACTIVATION_BUF_LEN4*DATA_DW-1:0] act_sr4,
    input [SRAM8192_AW-1:0] addr_decoder_w_init,    // from top.v, w0
    input [SRAM1024_AW-1:0] addr_decoder_scales_init,  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale
    input [SRAM1024_AW-1:0] addr_decoder_b_init,
    
    input [SRAM32_DW-1 : 0] sram1_dout,   //sram1_dout: b and scales
    input [SRAM16_DW-1 : 0] sram4_dout,
    input [SRAM16_DW-1 : 0] sram5_dout, 
    input [SRAM16_DW-1 : 0] sram6_dout,
    input layer_rdy, // 
    input [3-1 : 0] decoder_top_state,


    output [2*PE_NUM-1:0] shift_crl_all,
    
    output reg layer_out_vld,
    output reg [SRAM8192_AW-1:0] addr_sram,    // data width of weight and bias are the same, so no need to differenciate
    output sram1_en,
    output sram4_en,
    output sram5_en,
    output sram6_en,
    output reg signed [W_DW-1 : 0] decoder_w, //segment.v
    output reg signed [SCALE_DW -1 : 0] decoder_scale,
    output reg signed [SCALE_DW -1 : 0] decoder_b1,
    output reg signed [SCALE_DW -1 : 0] decoder_b2, // for top_decoder_state_16
    // output reg signed [SCALE_DW -1 : 0] decoder_b3_par, // for top_decoder_state_8
    // output reg signed [SCALE_DW -1 : 0] decoder_b4_par, // for top_decoder_state_8
    output layer_done,    
    // output reg reorder_en,

    output reg spad1_w_we_en, // the enable signal for the first spad
    output reg [PE_NUM-2:0] spad_w_we_en_2_32, //the enable signal for the rest of PEs
    output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_re, //pipeline
    output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_we,  // serial operation, so share one addr_we
    output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_a_addr_re, //pipeline
    output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_a_addr_we, //new
    output reg [PE_NUM-1 : 0] spad_a_we_en_1_32, //new
    output reg [INPUT_DW -1 : 0] spad1_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad2_a_data_sram_in, 
    output reg [DATA_DW -1 : 0] spad3_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad4_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad5_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad6_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad7_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad8_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad9_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad10_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad11_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad12_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad13_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad14_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad15_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad16_a_data_sram_in,  
    output reg [DATA_DW -1 : 0] spad17_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad18_a_data_sram_in, 
    output reg [DATA_DW -1 : 0] spad19_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad20_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad21_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad22_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad23_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad24_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad25_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad26_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad27_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad28_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad29_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad30_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad31_a_data_sram_in,
    output reg [DATA_DW -1 : 0] spad32_a_data_sram_in,  


    output  reg [1:0]  mult_a_crl, // 00:idle, 01:requantize     
    output  reg [1:0]  mult_b_crl,
    output reg [1:0] add_a_crl,
    output  [1:0] add_b_crl,

    output reg [3*PE_NUM-1:0] mult_int8_crl_all,
    output mult_out_round_en,
    // output sum_a_final_en,
    // output sum_b_final_en,

    //specially for cnn11
    output reg [$clog2(PE_NUM+1)-1:0] cnt_cho_32_3d,
    //specially for cnn22
    output cnn22_is_first,
    // specially for dcnn1
    output reg dcnn1_temp_value_vld,
    output dcnn1_transfer_temp_value_en,
    output dcnn1_temp_rst,
    output [PE_NUM-1:0] cnt_bt_all,
    output signed [DATA_BQ_DW-1 : 0]  dcnn1_temp_value_for_1,// 000:idle/reset, 001:gatess_mac, 011: transfer, 111:hold, 
    input  [SRAM8_DW-1:0]  sram_act_dout,
    output [SRAM8192_AW -1 : 0] addr_sram_act,
    output sram_act_en,
    output sram_act_we,
    output reg [SRAM8_DW-1:0]  sram_act_din,
    input signed [DATA_DW-1: 0] decoder_out, // dcnn1, cnn11, cnn12
    input [2*DATA_DW-1: 0] decoder_out_cat // cnn21, cnn22
    ); 

    localparam N_TOP = 3;
    localparam dcnn1 =  3'b001; //from sram
    localparam cnn11 = 3'b011;
    localparam cnn12 = 3'b111;
    localparam dcnn2 = 3'b101;
    localparam cnn21  = 3'b110 ;
    localparam cnn22 = 3'b010;

    localparam idle_top = 3'b000;
    // localparam reorder = 3'b001;
    localparam load_scale =  3'b011; //from sram
    localparam computate = 3'b111;
    localparam tail = 3'b110;
    localparam done_top = 3'b100;
    localparam wait_top = 3'b010;

    localparam TAIL_TIMES_32  = PE_NUM+2; // DCNN1, CNN11, CNN12
    localparam TAIL_TIMES_16 = PE_NUM/2+2;//DCNN2,
    localparam TAIL_TIMES_8 = PE_NUM/4+2; //CNN21, CNN22
    localparam BLOCK_TIMES = DCNN1_CHIN/PE_NUM; //2, the second layer
    localparam PAR_PES_DCNN2 = PE_NUM/DCNN2_CHIN;//2
    localparam PAR_PES_CNN2 = PE_NUM/CNN21_CHIN;//4
    
    reg [N_TOP-1:0] layer_top_state;
    reg [N_TOP-1:0] layer_top_state_next;

    reg computate_done;

    reg [$clog2(TAIL_TIMES_32+1)-1:0] cnt_tail;
    reg [$clog2(DCNN_KS+1)-1 : 0] cnt_ks;
    reg [$clog2(DCNN2_LENGTH_OUT+1)-1 : 0] cnt_lo;
    reg  cnt_bt;
    reg [$clog2(DCNN1_CHOUT+1)-1 : 0] cnt_cho;


    always @(*) begin
        case (decoder_top_state)
            dcnn1: begin
                computate_done =  ((layer_top_state == computate) & (cnt_ks == DCNN_KS) & (cnt_lo == DCNN1_LENGTH_OUT-1) & (cnt_cho == DCNN1_CHOUT-1) & (cnt_bt == 1));
            end
            cnn11:begin
                computate_done = ((layer_top_state == computate) & (cnt_ks == CNN_KS) & (cnt_lo == CNN11_LENGTH_OUT-1) & (cnt_cho == CNN11_CHOUT-1));
            end
            cnn12:begin
                computate_done = ((layer_top_state == computate) & (cnt_ks == CNN_KS) & (cnt_lo == CNN12_LENGTH_OUT-1) & (cnt_cho == CNN12_CHOUT-1));
            end
            dcnn2:begin
                computate_done = ((layer_top_state == computate) & (cnt_ks == DCNN_KS) & (cnt_lo == DCNN2_LENGTH_OUT-1) & (cnt_cho == DCNN2_CHOUT/PAR_PES_DCNN2-1));
            end
            cnn21:begin
                computate_done = ((layer_top_state == computate) & (cnt_ks == CNN_KS) & (cnt_lo == CNN21_LENGTH_OUT-1) & (cnt_cho == CNN21_CHOUT/PAR_PES_CNN2-1));
            end
            cnn22:begin
                computate_done = ((layer_top_state == computate) & (cnt_ks == CNN_KS) & (cnt_lo == CNN22_LENGTH_OUT-1) & (cnt_cho == CNN22_CHOUT/PAR_PES_CNN2-1));
            end
            default: computate_done = 0;
        endcase
    end
    // assign computate_done = ((layer_top_state == computate) & (cnt_ks == DCNN_KS) & (cnt_lo == DCNN1_LENGTH_OUT-1) & (cnt_cho == DCNN1_CHOUT-1) & (cnt_bt == 1));
    wire decoder_top_state_32;
    wire decoder_top_state_16;
    wire decoder_top_state_8;

    assign decoder_top_state_32 = (decoder_top_state == dcnn1) | (decoder_top_state == cnn11) | (decoder_top_state == cnn12);
    assign decoder_top_state_16 = (decoder_top_state == dcnn2);
    assign decoder_top_state_8 = (decoder_top_state == cnn21) | (decoder_top_state == cnn22);

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            layer_top_state <= idle_top;
        else
            layer_top_state <= layer_top_state_next;
    end

    always @(*) begin
        case(layer_top_state)
        idle_top: begin
            if (layer_rdy) layer_top_state_next = load_scale;
            else layer_top_state_next = idle_top;
        end
        load_scale: begin
            layer_top_state_next = computate;
        end
        computate: begin
            if (computate_done) layer_top_state_next = tail;
            else layer_top_state_next = computate;            
        end
        tail: begin
            if (decoder_top_state_32) begin
                if (cnt_tail == TAIL_TIMES_32) begin
                
                    layer_top_state_next = wait_top;
                end
                else layer_top_state_next = tail;                 
            end
            else if (decoder_top_state_16) begin
                if (cnt_tail == TAIL_TIMES_16) begin
                
                    layer_top_state_next = wait_top;
                end
                else layer_top_state_next = tail;                
            end
            else if (decoder_top_state_8) begin
                if (cnt_tail == TAIL_TIMES_8) begin
                
                    layer_top_state_next = wait_top;
                end
                else layer_top_state_next = tail;                
            end
            else layer_top_state_next = idle_top; 
           
        end
        wait_top: layer_top_state_next = done_top;
        done_top: layer_top_state_next = idle_top;
        default: layer_top_state_next = idle_top;
        endcase
        
    end
    assign layer_done = (layer_top_state==done_top);

    always@(posedge wclk or negedge rst_n) begin
        if (!rst_n) cnt_tail <= 0;
        else begin
            if (layer_top_state == tail) begin
                if (decoder_top_state_32) begin
                    cnt_tail <= (cnt_tail == TAIL_TIMES_32)? 0 : cnt_tail + 1;
                end
                else if (decoder_top_state_16) begin
                    cnt_tail <= (cnt_tail == TAIL_TIMES_16)? 0 : cnt_tail + 1;
                end
                else if (decoder_top_state_8) begin
                    cnt_tail <= (cnt_tail == TAIL_TIMES_8)? 0 : cnt_tail + 1;
                end
                else;
            end
        end
    end


   
    reg load_scale_end;

    always @(negedge sclk or negedge rst_n) begin
        if (!rst_n) begin
            decoder_scale <= 0;
            load_scale_end <= 0;
        end
        else begin
            if (layer_top_state == load_scale) begin
                if (!load_scale_end) begin
                    decoder_scale <=  sram1_dout;
                    load_scale_end <= 1;
                end
                else begin
                    decoder_scale  <= decoder_scale;
                    load_scale_end <= load_scale_end;                
                end
            end

            else begin
                decoder_scale  <= decoder_scale;
                load_scale_end <= 0;             
            end
        end
    end  




    // sub FSM
    localparam N = 3;
    localparam idle    = 3'b000;
    localparam load_w = 3'b010; // from sram
    localparam load_b = 3'b011;
    localparam load_a = 3'b111; // from sram
    localparam pe_rst = 3'b001;
    localparam mac   = 3'b110; // from sr
    localparam done    = 3'b101;    


    reg [N-1 : 0] conv_state;
    reg [N-1 : 0] conv_state_next;

    wire decoder_top_state_dcnn;
    assign decoder_top_state_dcnn = (decoder_top_state == dcnn1) | (decoder_top_state == dcnn2);
    wire decoder_top_state_cnn;
    assign decoder_top_state_cnn = ((decoder_top_state == cnn11) | (decoder_top_state == cnn12) |(decoder_top_state == cnn21) |(decoder_top_state == cnn22) );

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            conv_state <= idle;
        else
            conv_state <= conv_state_next;
    end
    reg  [N-1:0] conv_state_mem[PE_NUM-2:0];
    
    reg [(PE_NUM-1)*1-1:0] cnt_bt_mem;
    assign cnt_bt_all = {cnt_bt_mem, cnt_bt};
    integer i;
    integer j;
    // integer m;
    // always @(posedge wclk or negedge rst_n) begin
    //     if (!rst_n) begin 
    //         cnt_bt_mem <= 0;
    //         for (i = 0; i < PE_NUM-1; i = i+1) begin
    //             conv_state_mem[i] <= 0;
    //         end
    //     end
    //     else begin
    //         cnt_bt_mem[0] <= cnt_bt;
    //         conv_state_mem[0] <= conv_state;
    //         for (j = 1; j < PE_NUM-1; j = j+1) begin
    //             conv_state_mem[j] <= conv_state_mem[j-1];
    //             cnt_bt_mem[j] <= cnt_bt_mem[j-1];
    //         end
    //     end
    // end

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin 
            cnt_bt_mem <= 0;
            for (i = 0; i < PE_NUM-1; i = i+1) begin
                conv_state_mem[i] <= 0;
            end
        end
        else begin
            cnt_bt_mem[0] <= cnt_bt;
            conv_state_mem[0] <= conv_state;
            if (decoder_top_state_32) begin
                for (j = 1; j < PE_NUM-1; j = j+1) begin
                    conv_state_mem[j] <= conv_state_mem[j-1];
                    cnt_bt_mem[j] <= cnt_bt_mem[j-1];
                end
            end
            else if (decoder_top_state_16) begin
                conv_state_mem [1] <= conv_state_mem[0];
                conv_state_mem [2] <= conv_state_mem[1];
                conv_state_mem [3] <= conv_state_mem[2];
                conv_state_mem [4] <= conv_state_mem[3];
                conv_state_mem [5] <= conv_state_mem[4];
                conv_state_mem [6] <= conv_state_mem[5];
                conv_state_mem [7] <= conv_state_mem[6];
                conv_state_mem [8] <= conv_state_mem[7];
                conv_state_mem [9] <= conv_state_mem[8];
                conv_state_mem [10] <= conv_state_mem[9];
                conv_state_mem [11] <= conv_state_mem[10];
                conv_state_mem [12] <= conv_state_mem[11];
                conv_state_mem [13] <= conv_state_mem[12];
                conv_state_mem [14] <= conv_state_mem[13];
                conv_state_mem [15] <= 0;
                conv_state_mem [16] <= 0;
                conv_state_mem [17] <= 0;
                conv_state_mem [18] <= 0;
                conv_state_mem [19] <= 0;
                conv_state_mem [20] <= 0;
                conv_state_mem [21] <= 0;
                conv_state_mem [22] <= 0;
                conv_state_mem [23] <= 0;
                conv_state_mem [24] <= 0;
                conv_state_mem [25] <= 0;
                conv_state_mem [26] <= 0;
                conv_state_mem [27] <= 0;
                conv_state_mem [28] <= 0;
                conv_state_mem [29] <= 0;
                conv_state_mem [30] <= 0;   
                cnt_bt_mem[1] <= cnt_bt_mem[0];
                cnt_bt_mem[2] <= cnt_bt_mem[1];
                cnt_bt_mem[3] <= cnt_bt_mem[2];
                cnt_bt_mem[4] <= cnt_bt_mem[3];
                cnt_bt_mem[5] <= cnt_bt_mem[4];
                cnt_bt_mem[6] <= cnt_bt_mem[5];
                cnt_bt_mem[7] <= cnt_bt_mem[6];
                cnt_bt_mem[8] <= cnt_bt_mem[7];
                cnt_bt_mem[9] <= cnt_bt_mem[8];
                cnt_bt_mem[10] <= cnt_bt_mem[9];
                cnt_bt_mem[11] <= cnt_bt_mem[10];
                cnt_bt_mem[12] <= cnt_bt_mem[11];
                cnt_bt_mem[13] <= cnt_bt_mem[12];
                cnt_bt_mem[14] <= cnt_bt_mem[13];
                cnt_bt_mem[15] <= 0;
                cnt_bt_mem[16] <= 0;
                cnt_bt_mem[17] <= 0;
                cnt_bt_mem[18] <= 0;
                cnt_bt_mem[19] <= 0;
                cnt_bt_mem[20] <= 0;
                cnt_bt_mem[21] <= 0;
                cnt_bt_mem[22] <= 0;
                cnt_bt_mem[23] <= 0;
                cnt_bt_mem[24] <= 0;
                cnt_bt_mem[25] <= 0;
                cnt_bt_mem[26] <= 0;
                cnt_bt_mem[27] <= 0;
                cnt_bt_mem[28] <= 0;
                cnt_bt_mem[29] <= 0;
                cnt_bt_mem[30] <= 0;             
                // for (j = 1; j < PE_NUM/2-1; j = j+1) begin
                //     conv_state_mem[j] <= conv_state_mem[j-1];
                //     cnt_bt_mem[j] <= cnt_bt_mem[j-1];
                // end  
                // for (m = PE_NUM/2; m < PE_NUM/-1; m = m+1) begin
                //     conv_state_mem[m] <= 0;
                //     cnt_bt_mem[m] <= 0;
                // end                                
            end
            else begin
                conv_state_mem [1] <= conv_state_mem[0];
                conv_state_mem [2] <= conv_state_mem[1];
                conv_state_mem [3] <= conv_state_mem[2];
                conv_state_mem [4] <= conv_state_mem[3];
                conv_state_mem [5] <= conv_state_mem[4];
                conv_state_mem [6] <= conv_state_mem[5];
                conv_state_mem [7] <= 0;
                conv_state_mem [8] <= 0;
                conv_state_mem [9] <= 0;
                conv_state_mem [10] <= 0;
                conv_state_mem [11] <= 0;
                conv_state_mem [12] <= 0;
                conv_state_mem [13] <= 0;
                conv_state_mem [14] <= 0;
                conv_state_mem [15] <= 0;
                conv_state_mem [16] <= 0;
                conv_state_mem [17] <= 0;
                conv_state_mem [18] <= 0;
                conv_state_mem [19] <= 0;
                conv_state_mem [20] <= 0;
                conv_state_mem [21] <= 0;
                conv_state_mem [22] <= 0;
                conv_state_mem [23] <= 0;
                conv_state_mem [24] <= 0;
                conv_state_mem [25] <= 0;
                conv_state_mem [26] <= 0;
                conv_state_mem [27] <= 0;
                conv_state_mem [28] <= 0;
                conv_state_mem [29] <= 0;
                conv_state_mem [30] <= 0;
                cnt_bt_mem[1] <= cnt_bt_mem[0];
                cnt_bt_mem[2] <= cnt_bt_mem[1];
                cnt_bt_mem[3] <= cnt_bt_mem[2];
                cnt_bt_mem[4] <= cnt_bt_mem[3];
                cnt_bt_mem[5] <= cnt_bt_mem[4];
                cnt_bt_mem[6] <= cnt_bt_mem[5];
                cnt_bt_mem[7] <= 0;
                cnt_bt_mem[8] <= 0;
                cnt_bt_mem[9] <= 0;
                cnt_bt_mem[10] <= 0;
                cnt_bt_mem[11] <= 0;
                cnt_bt_mem[12] <= 0;
                cnt_bt_mem[13] <= 0;
                cnt_bt_mem[14] <= 0;
                cnt_bt_mem[15] <= 0;
                cnt_bt_mem[16] <= 0;
                cnt_bt_mem[17] <= 0;
                cnt_bt_mem[18] <= 0;
                cnt_bt_mem[19] <= 0;
                cnt_bt_mem[20] <= 0;
                cnt_bt_mem[21] <= 0;
                cnt_bt_mem[22] <= 0;
                cnt_bt_mem[23] <= 0;
                cnt_bt_mem[24] <= 0;
                cnt_bt_mem[25] <= 0;
                cnt_bt_mem[26] <= 0;
                cnt_bt_mem[27] <= 0;
                cnt_bt_mem[28] <= 0;
                cnt_bt_mem[29] <= 0;
                cnt_bt_mem[30] <= 0;

                // for (j = 1; j < PE_NUM/4-1; j = j+1) begin
                //     conv_state_mem[j] <= conv_state_mem[j-1];
                //     cnt_bt_mem[j] <= cnt_bt_mem[j-1];
                // end  
                // for (m = PE_NUM/4; m < PE_NUM/-1; m = m+1) begin
                //     conv_state_mem[m] <= 0;
                //     cnt_bt_mem[m] <= 0;
                // end                   
            end

        end
    end

wire cnt_bt_32 ;
assign cnt_bt_32 = cnt_bt_mem[PE_NUM-2];


reg [$clog2(DCNN1_CHOUT+1)-1:0] cnt_cho_mem [PE_NUM-2:0];

integer u1;
integer u2;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin 
        for (u1 = 0; u1 < PE_NUM-1; u1 = u1+1) begin
            cnt_cho_mem[u1] <= 0;
        end
    end
    else begin
        if (decoder_top_state  == cnn11) begin // output need to split into two srs
            cnt_cho_mem[0] <= cnt_cho;
            for (u2 = 1; u2 < PE_NUM-1; u2 = u2+1) begin
                cnt_cho_mem[u2] <= cnt_cho_mem[u2-1];
            end
        end
    end
end
wire [$clog2(PE_NUM+1)-1:0] cnt_cho_32;
assign cnt_cho_32 = cnt_cho_mem[PE_NUM-2];
reg [$clog2(PE_NUM+1)-1:0] cnt_cho_32_d;
reg [$clog2(PE_NUM+1)-1:0] cnt_cho_32_2d;

always@(posedge wclk or negedge rst_n)
begin
    if (!rst_n) begin
        cnt_cho_32_d <= 0;
        cnt_cho_32_2d <= 0;
        cnt_cho_32_3d <= 0;
    end
    else
    begin
        if (decoder_top_state == cnn11) begin
            cnt_cho_32_d <= cnt_cho_32;
            cnt_cho_32_2d <= cnt_cho_32_d;
            cnt_cho_32_3d <= cnt_cho_32_2d;
        end
        else begin
            cnt_cho_32_d <= 0;
            cnt_cho_32_2d <= 0;
            cnt_cho_32_3d <= 0;            
        end
    end
end
    always @(*) begin
        case (conv_state)
            idle: begin
                if (layer_top_state_next == computate) begin
                    conv_state_next = load_w;
                    // if (decoder_top_state_dcnn)
                    //     conv_state_next = load_w;
                    // else
                    //     conv_state_next = load_b;
                end
                else
                    conv_state_next = idle;
            end
            
            load_w: begin
                if (decoder_top_state_dcnn)
                    conv_state_next = load_a;
                else
                    conv_state_next = load_b;
                // conv_state_next = load_a;
            end
            load_b: conv_state_next = load_a;
            load_a: begin
                conv_state_next = mac;
            end
            // pe_rst:begin
            //     conv_state_next = mac;
            // end

            mac: begin
                if (decoder_top_state_dcnn) begin
                    if (cnt_ks != DCNN_KS)
                        conv_state_next = mac;
                    else conv_state_next = pe_rst;
                end
                else begin
                    if (cnt_ks != CNN_KS)
                        conv_state_next = mac;
                    else conv_state_next = pe_rst;                    
                end
            end 
            pe_rst: begin
                if (decoder_top_state  == dcnn1) begin 
                    if (cnt_lo != DCNN1_LENGTH_OUT-1) 
                        conv_state_next = load_a;
                    else begin
                        if (cnt_bt != BLOCK_TIMES-1)
                            conv_state_next = load_w;
                        else begin
                            if (cnt_cho != DCNN1_CHOUT-1)
                                conv_state_next = load_w;
                            else
                                conv_state_next = done;
                        end
                    end
                end
                else if (decoder_top_state  == cnn11) begin
                    if (cnt_lo != CNN11_LENGTH_OUT-1) 
                        conv_state_next = load_a;
                    else begin
                        if (cnt_cho != CNN11_CHOUT-1)
                            conv_state_next = load_w;
                        else
                            conv_state_next = done;
                    end                    
                end
                else if (decoder_top_state  == cnn12) begin
                    if (cnt_lo != CNN12_LENGTH_OUT-1) 
                        conv_state_next = load_a;
                    else begin
                        if (cnt_cho != CNN12_CHOUT-1)
                            conv_state_next = load_w;
                        else
                            conv_state_next = done;
                    end                      
                end
                else if (decoder_top_state  == dcnn2) begin
                    if (cnt_lo != DCNN2_LENGTH_OUT-1) 
                        conv_state_next = load_a;
                    else begin
                        if (cnt_cho != DCNN2_CHOUT/PAR_PES_DCNN2-1)
                            conv_state_next = load_w;
                        else
                            conv_state_next = done;
                    end                    
                end
                else if (decoder_top_state  == cnn21) begin
                    if (cnt_lo != CNN21_LENGTH_OUT-1) 
                        conv_state_next = load_a;
                    else begin
                        if (cnt_cho != CNN21_CHOUT/PAR_PES_CNN2-1)
                            conv_state_next = load_w;
                        else
                            conv_state_next = done;
                    end                     
                end
                else if (decoder_top_state == cnn22) begin
                    if (cnt_lo != CNN22_LENGTH_OUT-1) 
                        conv_state_next = load_a;
                    else begin
                        if (cnt_cho != CNN22_CHOUT/PAR_PES_CNN2-1)
                            conv_state_next = load_w;
                        else
                            conv_state_next = done;
                    end                     
                end
                else conv_state_next = idle;
              
            end               
            done: conv_state_next = idle;
            default: conv_state_next = idle;
        endcase
    end 
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            cnt_lo <=  0;
            cnt_ks <= 0;    
            cnt_bt <= 0;
            cnt_cho  <= 0;         
        end
        else begin
            if (conv_state == mac) begin
                if (decoder_top_state_dcnn) begin
                    if (cnt_ks == DCNN_KS)   begin //2 clk, one for transfer one for reset                                                                                                                                                                                                                                                                                                                                                                                                                                          ) begin
                        cnt_ks <= 0;
                    end
                    else cnt_ks <= cnt_ks + 1'b1;
                end
                else begin
                    if (cnt_ks == CNN_KS)   begin //2 clk, one for transfer one for reset                                                                                                                                                                                                                                                                                                                                                                                                                                          ) begin
                        cnt_ks <= 0;
                    end
                    else cnt_ks <= cnt_ks + 1'b1;                    
                end
            end
            else if (conv_state == pe_rst) begin
                if (decoder_top_state == dcnn1) begin
                    if (cnt_lo == DCNN1_LENGTH_OUT-1) begin
                        cnt_lo <= 0;
                        if (cnt_bt == BLOCK_TIMES-1) begin
                            cnt_bt <= 0;
                            if (cnt_cho == DCNN1_CHOUT-1) cnt_cho <= 0;
                            else cnt_cho <= cnt_cho + 1;
                        end
                        else cnt_bt <= cnt_bt + 1;
                    end
                    else
                        cnt_lo <= cnt_lo + 1;        
                end
                else if (decoder_top_state == cnn11) begin
                    if (cnt_lo == CNN11_LENGTH_OUT-1) begin
                        cnt_lo <= 0;
                        if (cnt_cho == CNN11_CHOUT-1) cnt_cho <= 0;
                        else cnt_cho <= cnt_cho + 1;
                    end
                    else
                        cnt_lo <= cnt_lo + 1;        

                end   
                else if (decoder_top_state==cnn12) begin
                    if (cnt_lo == CNN12_LENGTH_OUT-1) begin
                        cnt_lo <= 0;
                        if (cnt_cho == CNN12_CHOUT-1) cnt_cho <= 0;
                        else cnt_cho <= cnt_cho + 1;
                    end
                    else
                        cnt_lo <= cnt_lo + 1;                      
                end  
                else if (decoder_top_state == dcnn2) begin
                    if (cnt_lo == DCNN2_LENGTH_OUT-1) begin
                        cnt_lo <= 0;
                        if (cnt_cho == DCNN2_CHOUT/PAR_PES_DCNN2-1) cnt_cho <= 0;
                        else cnt_cho <= cnt_cho + 1;
                    end
                    else
                        cnt_lo <= cnt_lo + 1;                     
                end
                else if (decoder_top_state == cnn21) begin
                    if (cnt_lo == CNN21_LENGTH_OUT-1) begin
                        cnt_lo <= 0;
                        if (cnt_cho == CNN21_CHOUT/PAR_PES_CNN2-1) cnt_cho <= 0;
                        else cnt_cho <= cnt_cho + 1;
                    end
                    else
                        cnt_lo <= cnt_lo + 1;                     
                end                
                else if (decoder_top_state == cnn22) begin
                    if (cnt_lo == CNN22_LENGTH_OUT-1) begin
                        cnt_lo <= 0;
                        if (cnt_cho == CNN22_CHOUT/PAR_PES_CNN2-1) cnt_cho <= 0;
                        else cnt_cho <= cnt_cho + 1;
                    end
                    else
                        cnt_lo <= cnt_lo + 1;                     
                end  
                else;
            end
            else if (conv_state == idle) begin
                cnt_ks <= 0;    
                cnt_bt <= 0;
                cnt_lo <=  0;                                                                                                  
                cnt_cho  <= 0;             
            end
            else begin
                cnt_ks <= cnt_ks;    
                cnt_bt <= cnt_bt;
                cnt_lo <=  cnt_lo;
                cnt_cho  <= cnt_cho;
            end
        end
    end   

    reg [SRAM8192_AW-1:0] addr_decoder_w;


    reg  spad1_w_we_end;
    reg [PE_NUM-2:0] spad2_32_w_we_end;
    
    reg [SRAM1024_AW-1:0] addr_decoder_b;
    reg load_b1_end; // 
    reg load_b2_end; // for top_decoder_state_16 
    reg load_b3_end; // for top_decoder_state_32
    reg load_b4_end; // for top_decoder_state_32
    reg signed [SCALE_DW -1 : 0] decoder_b1_par; // for top_decoder_state_8
    reg signed [SCALE_DW -1 : 0] decoder_b2_par; // for top_decoder_state_8    
    reg signed [SCALE_DW -1 : 0] decoder_b3_par; // for top_decoder_state_8
    reg signed [SCALE_DW -1 : 0] decoder_b4_par; // for top_decoder_state_8
    reg signed [SCALE_DW -1 : 0] decoder_b3_par_d; // for top_decoder_state_8
    reg signed [SCALE_DW -1 : 0] decoder_b4_par_d; // for top_decoder_state_8

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            decoder_b3_par_d <= 0;
            decoder_b4_par_d <= 0;        
        end
        else begin
            decoder_b3_par_d <= decoder_b3_par;
            decoder_b4_par_d <= decoder_b4_par;
                     
        end
    end

    always @(negedge sclk or negedge rst_n) begin
        if (!rst_n) begin
            decoder_b1_par <= 0;
            decoder_b2_par <= 0;
            decoder_b3_par <= 0;
            decoder_b4_par <= 0;
            load_b1_end <= 0;
            load_b2_end <= 0;
            load_b3_end <= 0;
            load_b4_end <= 0;
        end
        else begin
            
            if (decoder_top_state_32) begin
                if (conv_state_mem[PE_NUM-2] == load_b) begin
                    if (!load_b1_end) begin
                        decoder_b1_par <=  sram1_dout;
                        load_b1_end <= 1;                        
                    end
                    else begin
                        decoder_b1_par <=  decoder_b1_par;
                        load_b1_end <= load_b1_end;                         
                    end
                end
                else begin
                    decoder_b1_par <=  decoder_b1_par;
                    load_b1_end <= 0;                     
                end
            end
            else if (decoder_top_state_8) begin
                if (conv_state_mem[PE_NUM/4-2] == load_b) begin
                    if (!load_b1_end) begin
                        decoder_b1_par <=  sram1_dout;
                        load_b1_end <= 1;                        
                    end
                    else if (!load_b2_end) begin
                        decoder_b2_par <=  sram1_dout;
                        load_b2_end <= 1; 
                    end
                    else if (!load_b3_end) begin
                        decoder_b3_par <=  sram1_dout;
                        load_b3_end <= 1;                         
                    end
                    else if (!load_b4_end) begin
                        decoder_b4_par <=  sram1_dout;
                        load_b4_end <= 1;                               
                    end
                    else begin
                        decoder_b1_par <=  decoder_b1_par;
                        decoder_b2_par<= decoder_b2_par;
                        decoder_b3_par <=  decoder_b3_par;
                        decoder_b4_par <= decoder_b4_par;                        
                        load_b1_end <= load_b1_end; 
                        load_b2_end   <= load_b2_end;  
                        load_b3_end <= load_b3_end; 
                        load_b4_end   <= load_b4_end;                                          
                    end
                end
                else begin
                    decoder_b1_par <=  decoder_b1_par;
                    decoder_b2_par<= decoder_b2_par;
                    decoder_b3_par <=  decoder_b3_par;
                    decoder_b4_par <= decoder_b4_par;                        
                    load_b1_end <= 0; 
                    load_b2_end   <= 0;  
                    load_b3_end <= 0; 
                    load_b4_end  <= 0;                   
                end                      
            end
            else;

        end
    end      
    always@(negedge sclk or negedge rst_n) begin
        if (!rst_n) begin
            addr_decoder_b <= 0;
        end
        else begin
            if (decoder_top_state_32) begin
                if (conv_state_mem[PE_NUM-2] == load_b) begin
                    if (!load_b1_end) addr_decoder_b <= addr_decoder_b + 1;
                    else addr_decoder_b <= addr_decoder_b;
                end
                else if (layer_top_state == idle_top) addr_decoder_b <= addr_decoder_b_init;
                else addr_decoder_b <= addr_decoder_b;
            end
            else if (decoder_top_state_8) begin
                if (conv_state_mem[PE_NUM/4-2] == load_b) begin
                    if (!load_b4_end) addr_decoder_b <= addr_decoder_b + 1;
                    else addr_decoder_b <= addr_decoder_b;
                end
                else if (layer_top_state == idle_top) addr_decoder_b <= addr_decoder_b_init;
                else addr_decoder_b <= addr_decoder_b;                
            end
            else addr_decoder_b <= addr_decoder_b;
        end
    end    
    /////////////////////////////////////need to modify//////////////////////////////////////////////

    // assign spad1_w_we_en = ((conv_state == load_w) & (!spad1_w_we_end))? 1:0; 
    // genvar k;
    // generate
    // for (k = 0; k < PE_NUM-1 ; k = k+1) begin:gen_spad_w_we_en
    //     assign spad_w_we_en_2_32[k] = ((conv_state_mem[k] == load_w)& (!spad2_32_w_we_end[k]))? 1:0;
    // end
    // endgenerate
    always @(*) begin
        if (decoder_top_state_32) begin
            spad1_w_we_en = ((conv_state == load_w) & (!spad1_w_we_end))? 1:0; 
            spad_w_we_en_2_32[0] =  ((conv_state_mem[0] == load_w)& (!spad2_32_w_we_end[0]))? 1:0;
            spad_w_we_en_2_32[1] =  ((conv_state_mem[1] == load_w)& (!spad2_32_w_we_end[1]))? 1:0;
            spad_w_we_en_2_32[2] =  ((conv_state_mem[2] == load_w)& (!spad2_32_w_we_end[2]))? 1:0;
            spad_w_we_en_2_32[3] =  ((conv_state_mem[3] == load_w)& (!spad2_32_w_we_end[3]))? 1:0;
            spad_w_we_en_2_32[4] =  ((conv_state_mem[4] == load_w)& (!spad2_32_w_we_end[4]))? 1:0;
            spad_w_we_en_2_32[5] =  ((conv_state_mem[5] == load_w)& (!spad2_32_w_we_end[5]))? 1:0;
            spad_w_we_en_2_32[6] =  ((conv_state_mem[6] == load_w)& (!spad2_32_w_we_end[6]))? 1:0;
            spad_w_we_en_2_32[7] =  ((conv_state_mem[7] == load_w)& (!spad2_32_w_we_end[7]))? 1:0;
            spad_w_we_en_2_32[8] =  ((conv_state_mem[8] == load_w)& (!spad2_32_w_we_end[8]))? 1:0;
            spad_w_we_en_2_32[9] =  ((conv_state_mem[9] == load_w)& (!spad2_32_w_we_end[9]))? 1:0;
            spad_w_we_en_2_32[10] =  ((conv_state_mem[10] == load_w)& (!spad2_32_w_we_end[10]))? 1:0;
            spad_w_we_en_2_32[11] =  ((conv_state_mem[11] == load_w)& (!spad2_32_w_we_end[11]))? 1:0;
            spad_w_we_en_2_32[12] =  ((conv_state_mem[12] == load_w)& (!spad2_32_w_we_end[12]))? 1:0;
            spad_w_we_en_2_32[13] =  ((conv_state_mem[13] == load_w)& (!spad2_32_w_we_end[13]))? 1:0;
            spad_w_we_en_2_32[14] =  ((conv_state_mem[14] == load_w)& (!spad2_32_w_we_end[14]))? 1:0;
            spad_w_we_en_2_32[15] =  ((conv_state_mem[15] == load_w)& (!spad2_32_w_we_end[15]))? 1:0;
            spad_w_we_en_2_32[16] =  ((conv_state_mem[16] == load_w)& (!spad2_32_w_we_end[16]))? 1:0;
            spad_w_we_en_2_32[17] =  ((conv_state_mem[17] == load_w)& (!spad2_32_w_we_end[17]))? 1:0;
            spad_w_we_en_2_32[18] =  ((conv_state_mem[18] == load_w)& (!spad2_32_w_we_end[18]))? 1:0;
            spad_w_we_en_2_32[19] =  ((conv_state_mem[19] == load_w)& (!spad2_32_w_we_end[19]))? 1:0;
            spad_w_we_en_2_32[20] =  ((conv_state_mem[20] == load_w)& (!spad2_32_w_we_end[20]))? 1:0;
            spad_w_we_en_2_32[21] =  ((conv_state_mem[21] == load_w)& (!spad2_32_w_we_end[21]))? 1:0;
            spad_w_we_en_2_32[22] =  ((conv_state_mem[22] == load_w)& (!spad2_32_w_we_end[22]))? 1:0;
            spad_w_we_en_2_32[23] =  ((conv_state_mem[23] == load_w)& (!spad2_32_w_we_end[23]))? 1:0;
            spad_w_we_en_2_32[24] =  ((conv_state_mem[24] == load_w)& (!spad2_32_w_we_end[24]))? 1:0;
            spad_w_we_en_2_32[25] =  ((conv_state_mem[25] == load_w)& (!spad2_32_w_we_end[25]))? 1:0;
            spad_w_we_en_2_32[26] =  ((conv_state_mem[26] == load_w)& (!spad2_32_w_we_end[26]))? 1:0;
            spad_w_we_en_2_32[27] =  ((conv_state_mem[27] == load_w)& (!spad2_32_w_we_end[27]))? 1:0;
            spad_w_we_en_2_32[28] =  ((conv_state_mem[28] == load_w)& (!spad2_32_w_we_end[28]))? 1:0;
            spad_w_we_en_2_32[29] =  ((conv_state_mem[29] == load_w)& (!spad2_32_w_we_end[29]))? 1:0;
            spad_w_we_en_2_32[30] =  ((conv_state_mem[30] == load_w)& (!spad2_32_w_we_end[30]))? 1:0;

        end
        else if (decoder_top_state_16) begin
            spad1_w_we_en = ((conv_state == load_w) & (!spad1_w_we_end))? 1:0; 
            spad_w_we_en_2_32[0] =  ((conv_state_mem[0] == load_w)& (!spad2_32_w_we_end[0]))? 1:0;
            spad_w_we_en_2_32[1] =  ((conv_state_mem[1] == load_w)& (!spad2_32_w_we_end[1]))? 1:0;
            spad_w_we_en_2_32[2] =  ((conv_state_mem[2] == load_w)& (!spad2_32_w_we_end[2]))? 1:0;
            spad_w_we_en_2_32[3] =  ((conv_state_mem[3] == load_w)& (!spad2_32_w_we_end[3]))? 1:0;
            spad_w_we_en_2_32[4] =  ((conv_state_mem[4] == load_w)& (!spad2_32_w_we_end[4]))? 1:0;
            spad_w_we_en_2_32[5] =  ((conv_state_mem[5] == load_w)& (!spad2_32_w_we_end[5]))? 1:0;
            spad_w_we_en_2_32[6] =  ((conv_state_mem[6] == load_w)& (!spad2_32_w_we_end[6]))? 1:0;
            spad_w_we_en_2_32[7] =  ((conv_state_mem[7] == load_w)& (!spad2_32_w_we_end[7]))? 1:0;
            spad_w_we_en_2_32[8] =  ((conv_state_mem[8] == load_w)& (!spad2_32_w_we_end[8]))? 1:0;
            spad_w_we_en_2_32[9] =  ((conv_state_mem[9] == load_w)& (!spad2_32_w_we_end[9]))? 1:0;
            spad_w_we_en_2_32[10] =  ((conv_state_mem[10] == load_w)& (!spad2_32_w_we_end[10]))? 1:0;
            spad_w_we_en_2_32[11] =  ((conv_state_mem[11] == load_w)& (!spad2_32_w_we_end[11]))? 1:0;
            spad_w_we_en_2_32[12] =  ((conv_state_mem[12] == load_w)& (!spad2_32_w_we_end[12]))? 1:0;
            spad_w_we_en_2_32[13] =  ((conv_state_mem[13] == load_w)& (!spad2_32_w_we_end[13]))? 1:0;
            spad_w_we_en_2_32[14] =  ((conv_state_mem[14] == load_w)& (!spad2_32_w_we_end[14]))? 1:0;
            spad_w_we_en_2_32[15] =  ((conv_state == load_w) & (spad1_w_we_end) & (!spad2_32_w_we_end[15]))? 1:0;
            spad_w_we_en_2_32[16] =  ((conv_state_mem[0] == load_w)& (spad2_32_w_we_end[0])& (!spad2_32_w_we_end[16]))? 1:0;
            spad_w_we_en_2_32[17] =  ((conv_state_mem[1] == load_w)& (spad2_32_w_we_end[1])& (!spad2_32_w_we_end[17]))? 1:0;
            spad_w_we_en_2_32[18] =  ((conv_state_mem[2] == load_w)& (spad2_32_w_we_end[2])& (!spad2_32_w_we_end[18]))? 1:0;
            spad_w_we_en_2_32[19] =  ((conv_state_mem[3] == load_w)& (spad2_32_w_we_end[3])& (!spad2_32_w_we_end[19]))? 1:0;
            spad_w_we_en_2_32[20] =  ((conv_state_mem[4] == load_w)& (spad2_32_w_we_end[4])& (!spad2_32_w_we_end[20]))? 1:0;
            spad_w_we_en_2_32[21] =  ((conv_state_mem[5] == load_w)& (spad2_32_w_we_end[5])& (!spad2_32_w_we_end[21]))? 1:0;
            spad_w_we_en_2_32[22] =  ((conv_state_mem[6] == load_w)& (spad2_32_w_we_end[6])& (!spad2_32_w_we_end[22]))? 1:0;
            spad_w_we_en_2_32[23] =  ((conv_state_mem[7] == load_w)& (spad2_32_w_we_end[7])& (!spad2_32_w_we_end[23]))? 1:0;
            spad_w_we_en_2_32[24] =  ((conv_state_mem[8] == load_w)& (spad2_32_w_we_end[8])& (!spad2_32_w_we_end[24]))? 1:0;
            spad_w_we_en_2_32[25] =  ((conv_state_mem[9] == load_w)& (spad2_32_w_we_end[9])& (!spad2_32_w_we_end[25]))? 1:0;
            spad_w_we_en_2_32[26] =  ((conv_state_mem[10] == load_w)& (spad2_32_w_we_end[10])& (!spad2_32_w_we_end[26]))? 1:0;
            spad_w_we_en_2_32[27] =  ((conv_state_mem[11] == load_w)& (spad2_32_w_we_end[11])& (!spad2_32_w_we_end[27]))? 1:0;
            spad_w_we_en_2_32[28] =  ((conv_state_mem[12] == load_w)& (spad2_32_w_we_end[12])& (!spad2_32_w_we_end[28]))? 1:0;
            spad_w_we_en_2_32[29] =  ((conv_state_mem[13] == load_w)& (spad2_32_w_we_end[13])& (!spad2_32_w_we_end[29]))? 1:0;
            spad_w_we_en_2_32[30] =  ((conv_state_mem[14] == load_w)& (spad2_32_w_we_end[14])& (!spad2_32_w_we_end[30]))? 1:0;            
        end
        else if (decoder_top_state_8) begin
            spad1_w_we_en = ((conv_state == load_w) & (!spad1_w_we_end))? 1:0; 
            spad_w_we_en_2_32[0] =  ((conv_state_mem[0] == load_w)& (!spad2_32_w_we_end[0]))? 1:0;
            spad_w_we_en_2_32[1] =  ((conv_state_mem[1] == load_w)& (!spad2_32_w_we_end[1]))? 1:0;
            spad_w_we_en_2_32[2] =  ((conv_state_mem[2] == load_w)& (!spad2_32_w_we_end[2]))? 1:0;
            spad_w_we_en_2_32[3] =  ((conv_state_mem[3] == load_w)& (!spad2_32_w_we_end[3]))? 1:0;
            spad_w_we_en_2_32[4] =  ((conv_state_mem[4] == load_w)& (!spad2_32_w_we_end[4]))? 1:0;
            spad_w_we_en_2_32[5] =  ((conv_state_mem[5] == load_w)& (!spad2_32_w_we_end[5]))? 1:0;
            spad_w_we_en_2_32[6] =  ((conv_state_mem[6] == load_w)& (!spad2_32_w_we_end[6]))? 1:0;
            
            spad_w_we_en_2_32[7] =  ((conv_state == load_w) & (spad1_w_we_end) & (!spad2_32_w_we_end[7]))? 1:0;
            spad_w_we_en_2_32[8] =  ((conv_state_mem[0] == load_w) & (spad2_32_w_we_end[0]) & (!spad2_32_w_we_end[8]))? 1:0;   
            spad_w_we_en_2_32[9] =  ((conv_state_mem[1] == load_w) & (spad2_32_w_we_end[1]) & (!spad2_32_w_we_end[9]))? 1:0;   
            spad_w_we_en_2_32[10] =  ((conv_state_mem[2] == load_w) & (spad2_32_w_we_end[2]) & (!spad2_32_w_we_end[10]))? 1:0;   
            spad_w_we_en_2_32[11] =  ((conv_state_mem[3] == load_w) & (spad2_32_w_we_end[3]) & (!spad2_32_w_we_end[11]))? 1:0;   
            spad_w_we_en_2_32[12] =  ((conv_state_mem[4] == load_w) & (spad2_32_w_we_end[4]) & (!spad2_32_w_we_end[12]))? 1:0;   
            spad_w_we_en_2_32[13] =  ((conv_state_mem[5] == load_w) & (spad2_32_w_we_end[5]) & (!spad2_32_w_we_end[13]))? 1:0;   
            spad_w_we_en_2_32[14] =  ((conv_state_mem[6] == load_w) & (spad2_32_w_we_end[6]) & (!spad2_32_w_we_end[14]))? 1:0;  
            
            spad_w_we_en_2_32[15] =  ((conv_state == load_w) & (spad1_w_we_end) &(spad2_32_w_we_end[7]) & (!spad2_32_w_we_end[15]))? 1:0; 
            spad_w_we_en_2_32[16] =  ((conv_state_mem[0] == load_w) & (spad2_32_w_we_end[0]) &(spad2_32_w_we_end[8]) & (!spad2_32_w_we_end[16]))? 1:0; 
            spad_w_we_en_2_32[17] =  ((conv_state_mem[1] == load_w) & (spad2_32_w_we_end[1]) &(spad2_32_w_we_end[9]) & (!spad2_32_w_we_end[17]))? 1:0; 
            spad_w_we_en_2_32[18] =  ((conv_state_mem[2] == load_w) & (spad2_32_w_we_end[2]) &(spad2_32_w_we_end[10]) & (!spad2_32_w_we_end[18]))? 1:0; 
            spad_w_we_en_2_32[19] =  ((conv_state_mem[3] == load_w) & (spad2_32_w_we_end[3]) &(spad2_32_w_we_end[11]) & (!spad2_32_w_we_end[19]))? 1:0; 
            spad_w_we_en_2_32[20] =  ((conv_state_mem[4] == load_w) & (spad2_32_w_we_end[4]) &(spad2_32_w_we_end[12]) & (!spad2_32_w_we_end[20]))? 1:0; 
            spad_w_we_en_2_32[21] =  ((conv_state_mem[5] == load_w) & (spad2_32_w_we_end[5]) &(spad2_32_w_we_end[13]) & (!spad2_32_w_we_end[21]))? 1:0; 
            spad_w_we_en_2_32[22] =  ((conv_state_mem[6] == load_w) & (spad2_32_w_we_end[6]) &(spad2_32_w_we_end[14]) & (!spad2_32_w_we_end[22]))? 1:0; 

            spad_w_we_en_2_32[23] =  ((conv_state == load_w) & (spad1_w_we_end) &(spad2_32_w_we_end[7]) & (spad2_32_w_we_end[15])& (!spad2_32_w_we_end[23]))? 1:0;
            spad_w_we_en_2_32[24] =  ((conv_state_mem[0] == load_w) & (spad2_32_w_we_end[0]) &(spad2_32_w_we_end[8]) &(spad2_32_w_we_end[16]) & (!spad2_32_w_we_end[24]))? 1:0; 
            spad_w_we_en_2_32[25] =  ((conv_state_mem[1] == load_w) & (spad2_32_w_we_end[1]) &(spad2_32_w_we_end[9]) &(spad2_32_w_we_end[17]) & (!spad2_32_w_we_end[25]))? 1:0; 
            spad_w_we_en_2_32[26] =  ((conv_state_mem[2] == load_w) & (spad2_32_w_we_end[2]) &(spad2_32_w_we_end[10]) &(spad2_32_w_we_end[18]) & (!spad2_32_w_we_end[26]))? 1:0; 
            spad_w_we_en_2_32[27] =  ((conv_state_mem[3] == load_w) & (spad2_32_w_we_end[3]) &(spad2_32_w_we_end[11]) &(spad2_32_w_we_end[19]) & (!spad2_32_w_we_end[27]))? 1:0; 
            spad_w_we_en_2_32[28] =  ((conv_state_mem[4] == load_w) & (spad2_32_w_we_end[4]) &(spad2_32_w_we_end[12]) &(spad2_32_w_we_end[20]) & (!spad2_32_w_we_end[28]))? 1:0; 
            spad_w_we_en_2_32[29] =  ((conv_state_mem[5] == load_w) & (spad2_32_w_we_end[5]) &(spad2_32_w_we_end[13]) &(spad2_32_w_we_end[21]) & (!spad2_32_w_we_end[29]))? 1:0; 
            spad_w_we_en_2_32[30] =  ((conv_state_mem[6] == load_w) & (spad2_32_w_we_end[6]) &(spad2_32_w_we_end[14]) &(spad2_32_w_we_end[22]) & (!spad2_32_w_we_end[30]))? 1:0; 
        end
        else begin
            spad_w_we_en_2_32 = 0;
            spad1_w_we_en = 0;
        end
    end
    reg is_lsb;
    wire spad_w_addr_we_end;
    assign spad_w_addr_we_end = (decoder_top_state_dcnn)?( (spad_w_addr_we == DCNN_KS-1) ? 1:0):( (spad_w_addr_we == CNN_KS-1) ? 1:0);
    /////////////////////////////////////need to modify//////////////////////////////////////////////
    // integer a;
    // reg spad1_w_we_end_reg;
    // reg [PE_NUM-1:0] spad2_32_w_we_end_reg;
    // always @(negedge sclk or negedge rst_n) begin
    //     if (!rst_n)begin
    //         spad1_w_we_end_reg <= 0;
    //         spad2_32_w_we_end_reg <= 0;
    //     end
    //     else begin
    //         spad1_w_we_end_reg <= spad1_w_we_end;
    //         spad2_32_w_we_end_reg <= spad2_32_w_we_end;
    //     end
    // end
    
    always @(negedge sclk or negedge rst_n) begin
        if (!rst_n) begin
            spad1_w_we_end <= 0;
            spad2_32_w_we_end <= 0;
            spad_w_addr_we <= 0;
            addr_decoder_w <= 0;
        end
        else begin
            if (decoder_top_state_32) begin
                if (conv_state == load_w)  begin               
                    if (!spad1_w_we_end)  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad1_w_we_end <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad1_w_we_end <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end

                end
                // for (a =0; a < PE_NUM -1 ; a = a + 1) begin
                else if (conv_state_mem[0] == load_w)begin
                    if (!spad2_32_w_we_end[0]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[0] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[0] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[1] == load_w)begin
                    if (!spad2_32_w_we_end[1]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[1] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[1] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[2] == load_w)begin
                    if (!spad2_32_w_we_end[2]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[2] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[2] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[3] == load_w)begin
                    if (!spad2_32_w_we_end[3]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[3] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[3] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[4] == load_w)begin
                    if (!spad2_32_w_we_end[4]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[4] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[4] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[5] == load_w)begin
                    if (!spad2_32_w_we_end[5]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[5] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[5] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[6] == load_w)begin
                    if (!spad2_32_w_we_end[6]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[6] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[6] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[7] == load_w)begin
                    if (!spad2_32_w_we_end[7]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[7] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[7] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[8] == load_w)begin
                    if (!spad2_32_w_we_end[8]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[8] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[8] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[9] == load_w)begin
                    if (!spad2_32_w_we_end[9]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[9] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[9] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[10] == load_w)begin
                    if (!spad2_32_w_we_end[10]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[10] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[10] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[11] == load_w)begin
                    if (!spad2_32_w_we_end[11]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[11] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[11] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[12] == load_w)begin
                    if (!spad2_32_w_we_end[12]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[12] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[12] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[13] == load_w)begin
                    if (!spad2_32_w_we_end[13]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[13] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[13] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[14] == load_w)begin
                    if (!spad2_32_w_we_end[14]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[14] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[14] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[15] == load_w)begin
                    if (!spad2_32_w_we_end[15]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[15] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[15] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[16] == load_w)begin
                    if (!spad2_32_w_we_end[16]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[16] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[16] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[17] == load_w)begin
                    if (!spad2_32_w_we_end[17]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[17] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[17] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[18] == load_w)begin
                    if (!spad2_32_w_we_end[18]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[18] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[18] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[19] == load_w)begin
                    if (!spad2_32_w_we_end[19]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[19] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[19] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[20] == load_w)begin
                    if (!spad2_32_w_we_end[20]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[20] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[20] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[21] == load_w)begin
                    if (!spad2_32_w_we_end[21]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[21] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[21] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[22] == load_w)begin
                    if (!spad2_32_w_we_end[22]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[22] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[22] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[23] == load_w)begin
                    if (!spad2_32_w_we_end[23]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[23] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[23] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[24] == load_w)begin
                    if (!spad2_32_w_we_end[24]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[24] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[24] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[25] == load_w)begin
                    if (!spad2_32_w_we_end[25]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[25] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[25] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[26] == load_w)begin
                    if (!spad2_32_w_we_end[26]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[26] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[26] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[27] == load_w)begin
                    if (!spad2_32_w_we_end[27]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[27] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[27] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[28] == load_w)begin
                    if (!spad2_32_w_we_end[28]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[28] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[28] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[29] == load_w)begin
                    if (!spad2_32_w_we_end[29]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[29] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[29] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end
                else if (conv_state_mem[30] == load_w)begin
                    if (!spad2_32_w_we_end[30]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[30] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[30] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;                        
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w ;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad2_32_w_we_end <= spad2_32_w_we_end;
                    end
                end

            
                else if (layer_top_state == idle_top) begin
                    addr_decoder_w <= addr_decoder_w_init;// wait for the next time
                    spad1_w_we_end <= 0; // RESET
                    spad2_32_w_we_end <= 0;// RESET
                    spad_w_addr_we <= spad_w_addr_we;
                end
                else begin
                    addr_decoder_w <= addr_decoder_w ;
                    spad1_w_we_end <= 0; // RESET
                    spad2_32_w_we_end <= 0;// RESET
                    spad_w_addr_we <= spad_w_addr_we;
                end
            end
            else if (decoder_top_state_16) begin
                if (conv_state == load_w)  begin               
                    if (!spad1_w_we_end)  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad1_w_we_end <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad1_w_we_end <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[15]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[15] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[15] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end                        
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end   
                else if (conv_state_mem[0] == load_w) begin
                    if (!spad2_32_w_we_end[0])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[0] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[0] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[16]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[16] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[16] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end 
                else if (conv_state_mem[1] == load_w) begin
                    if (!spad2_32_w_we_end[1])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[1] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[1] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[17]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[17] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[17] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[2] == load_w) begin
                    if (!spad2_32_w_we_end[2])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[2] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[2] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[18]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[18] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[18] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[3] == load_w) begin
                    if (!spad2_32_w_we_end[3])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[3] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[3] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[19]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[19] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[19] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[4] == load_w) begin
                    if (!spad2_32_w_we_end[4])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[4] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[4] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[20]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[20] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[20] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[5] == load_w) begin
                    if (!spad2_32_w_we_end[5])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[5] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[5] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[21]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[21] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[21] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[6] == load_w) begin
                    if (!spad2_32_w_we_end[6])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[6] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[6] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[22]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[22] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[22] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[7] == load_w) begin
                    if (!spad2_32_w_we_end[7])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[7] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[7] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[23]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[23] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[23] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[8] == load_w) begin
                    if (!spad2_32_w_we_end[8])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[8] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[8] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[24]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[24] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[24] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[9] == load_w) begin
                    if (!spad2_32_w_we_end[9])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[9] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[9] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[25]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[25] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[25] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[10] == load_w) begin
                    if (!spad2_32_w_we_end[10])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[10] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[10] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[26]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[26] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[26] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[11] == load_w) begin
                    if (!spad2_32_w_we_end[11])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[11] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[11] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[27]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[27] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[27] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[12] == load_w) begin
                    if (!spad2_32_w_we_end[12])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[12] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[12] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[28]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[28] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[28] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[13] == load_w) begin
                    if (!spad2_32_w_we_end[13])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[13] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[13] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[29]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[29] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[29] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[14] == load_w) begin
                    if (!spad2_32_w_we_end[14])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[14] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[14] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[30]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[30] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[30] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end

                else if (layer_top_state == idle_top) begin
                    addr_decoder_w <= addr_decoder_w_init;// wait for the next time
                    spad1_w_we_end <= 0; // RESET
                    spad2_32_w_we_end <= 0;// RESET
                    spad_w_addr_we <= spad_w_addr_we;
                end
                else begin
                    addr_decoder_w <= addr_decoder_w ;
                    spad1_w_we_end <= 0; // RESET
                    spad2_32_w_we_end <= 0;// RESET
                    spad_w_addr_we <= spad_w_addr_we;
                end    
            end
            else if (decoder_top_state_8) begin
                if (conv_state == load_w)  begin               
                    if (!spad1_w_we_end)  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad1_w_we_end <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad1_w_we_end <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[7]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[7] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[7] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end                        
                    end
                    else if (!spad2_32_w_we_end[15]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[15] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[15] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end                        
                    end
                    else if (!spad2_32_w_we_end[23]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[23] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[23] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end                        
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end 
                else if (conv_state_mem[0] == load_w) begin
                    if (!spad2_32_w_we_end[0])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[0] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[0] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[8]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[8] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[8] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[16]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[16] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[16] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[24]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[24] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[24] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[1] == load_w) begin
                    if (!spad2_32_w_we_end[1])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[1] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[1] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[9]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[9] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[9] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[17]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[17] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[17] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[25]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[25] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[25] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[2] == load_w) begin
                    if (!spad2_32_w_we_end[2])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[2] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[2] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[10]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[10] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[10] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[18]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[18] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[18] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[26]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[26] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[26] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[3] == load_w) begin
                    if (!spad2_32_w_we_end[3])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[3] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[3] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[11]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[11] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[11] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[19]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[19] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[19] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[27]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[27] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[27] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[4] == load_w) begin
                    if (!spad2_32_w_we_end[4])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[4] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[4] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[12]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[12] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[12] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[20]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[20] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[20] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[28]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[28] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[28] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[5] == load_w) begin
                    if (!spad2_32_w_we_end[5])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[5] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[5] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[13]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[13] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[13] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[21]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[21] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[21] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[29]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[29] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[29] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (conv_state_mem[6] == load_w) begin
                    if (!spad2_32_w_we_end[6])  begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[6] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[6] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[14]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[14] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[14] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[22]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[22] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[22] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else if (!spad2_32_w_we_end[30]) begin
                        addr_decoder_w <= (~is_lsb)?addr_decoder_w + 1:addr_decoder_w;
                        if (spad_w_addr_we_end) begin
                            spad2_32_w_we_end[30] <= 1;
                            spad_w_addr_we <= 0;
                        end
                        else begin
                            spad2_32_w_we_end[30] <= 0;
                            spad_w_addr_we <= spad_w_addr_we + 1;
                        end
                    end
                    else begin
                        addr_decoder_w <= addr_decoder_w;
                        spad_w_addr_we <= spad_w_addr_we;
                        spad1_w_we_end <= spad1_w_we_end;
                    end
                end
                else if (layer_top_state == idle_top) begin
                    addr_decoder_w <= addr_decoder_w_init;// wait for the next time
                    spad1_w_we_end <= 0; // RESET
                    spad2_32_w_we_end <= 0;// RESET
                    spad_w_addr_we <= spad_w_addr_we;
                end
                else begin
                    addr_decoder_w <= addr_decoder_w ;
                    spad1_w_we_end <= 0; // RESET
                    spad2_32_w_we_end <= 0;// RESET
                    spad_w_addr_we <= spad_w_addr_we;
                end
            end
        end        
    end
    // integer s;
    
    always @(negedge sclk or negedge rst_n) begin
        if (!rst_n) is_lsb <= 1;
        else begin
            if (decoder_top_state_32) begin
                if (conv_state == load_w) begin
                    if (spad1_w_we_end) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                // for (s =0; s < PE_NUM -1 ; s = s + 1) begin
                else if (conv_state_mem[0] == load_w) begin
                    if (spad2_32_w_we_end[0]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[1] == load_w) begin
                    if (spad2_32_w_we_end[1]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[2] == load_w) begin
                    if (spad2_32_w_we_end[2]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[3] == load_w) begin
                    if (spad2_32_w_we_end[3]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[4] == load_w) begin
                    if (spad2_32_w_we_end[4]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[5] == load_w) begin
                    if (spad2_32_w_we_end[5]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[6] == load_w) begin
                    if (spad2_32_w_we_end[6]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[7] == load_w) begin
                    if (spad2_32_w_we_end[7]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[8] == load_w) begin
                    if (spad2_32_w_we_end[8]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[9] == load_w) begin
                    if (spad2_32_w_we_end[9]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[10] == load_w) begin
                    if (spad2_32_w_we_end[10]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[11] == load_w) begin
                    if (spad2_32_w_we_end[11]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[12] == load_w) begin
                    if (spad2_32_w_we_end[12]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[13] == load_w) begin
                    if (spad2_32_w_we_end[13]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[14] == load_w) begin
                    if (spad2_32_w_we_end[14]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[15] == load_w) begin
                    if (spad2_32_w_we_end[15]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[16] == load_w) begin
                    if (spad2_32_w_we_end[16]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[17] == load_w) begin
                    if (spad2_32_w_we_end[17]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[18] == load_w) begin
                    if (spad2_32_w_we_end[18]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[19] == load_w) begin
                    if (spad2_32_w_we_end[19]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[20] == load_w) begin
                    if (spad2_32_w_we_end[20]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[21] == load_w) begin
                    if (spad2_32_w_we_end[21]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[22] == load_w) begin
                    if (spad2_32_w_we_end[22]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[23] == load_w) begin
                    if (spad2_32_w_we_end[23]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[24] == load_w) begin
                    if (spad2_32_w_we_end[24]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[25] == load_w) begin
                    if (spad2_32_w_we_end[25]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[26] == load_w) begin
                    if (spad2_32_w_we_end[26]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[27] == load_w) begin
                    if (spad2_32_w_we_end[27]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[28] == load_w) begin
                    if (spad2_32_w_we_end[28]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[29] == load_w) begin
                    if (spad2_32_w_we_end[29]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end
                else if (conv_state_mem[30] == load_w) begin
                    if (spad2_32_w_we_end[30]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end         
                // end
                else begin
                    is_lsb <= 1;
                end
            end
            else if (decoder_top_state_16) begin
                if (conv_state == load_w) begin
                    if (spad2_32_w_we_end[15]) is_lsb <= is_lsb;
                    else  is_lsb <= ~is_lsb;
                end  
				else if (conv_state_mem[0] == load_w) begin
					if (spad2_32_w_we_end[16]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[1] == load_w) begin
					if (spad2_32_w_we_end[17]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[2] == load_w) begin
					if (spad2_32_w_we_end[18]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[3] == load_w) begin
					if (spad2_32_w_we_end[19]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[4] == load_w) begin
					if (spad2_32_w_we_end[20]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[5] == load_w) begin
					if (spad2_32_w_we_end[21]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[6] == load_w) begin
					if (spad2_32_w_we_end[22]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[7] == load_w) begin
					if (spad2_32_w_we_end[23]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[8] == load_w) begin
					if (spad2_32_w_we_end[24]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[9] == load_w) begin
					if (spad2_32_w_we_end[25]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[10] == load_w) begin
					if (spad2_32_w_we_end[26]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[11] == load_w) begin
					if (spad2_32_w_we_end[27]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[12] == load_w) begin
					if (spad2_32_w_we_end[28]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[13] == load_w) begin
					if (spad2_32_w_we_end[29]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[14] == load_w) begin
					if (spad2_32_w_we_end[30]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
                else begin
                    is_lsb <= 1;
                end

              
            end
            else if (decoder_top_state_8) begin
 				if (conv_state == load_w) begin
					if (spad2_32_w_we_end[23]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end               
				else if (conv_state_mem[0] == load_w) begin
					if (spad2_32_w_we_end[24]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[1] == load_w) begin
					if (spad2_32_w_we_end[25]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[2] == load_w) begin
					if (spad2_32_w_we_end[26]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[3] == load_w) begin
					if (spad2_32_w_we_end[27]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[4] == load_w) begin
					if (spad2_32_w_we_end[28]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[5] == load_w) begin
					if (spad2_32_w_we_end[29]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
				else if (conv_state_mem[6] == load_w) begin
					if (spad2_32_w_we_end[30]) is_lsb <= is_lsb;
					else  is_lsb <= ~is_lsb;
				end
                else begin
                    is_lsb <= 1;
                end
             
            end
            else begin
                is_lsb <= 1;                
            end
        end
    end


    reg load_w_time;
    always @(*) begin
        if (decoder_top_state_32) begin
            load_w_time = ((conv_state == load_w) |
                                (conv_state_mem[0] == load_w)|
                                (conv_state_mem[1] == load_w)|
                                (conv_state_mem[2] == load_w)|
                                (conv_state_mem[3] == load_w)|
                                (conv_state_mem[4] == load_w)|
                                (conv_state_mem[5] == load_w)|
                                (conv_state_mem[6] == load_w)|
                                (conv_state_mem[7] == load_w)|
                                (conv_state_mem[8] == load_w)|
                                (conv_state_mem[9] == load_w)|
                                (conv_state_mem[10] == load_w)|
                                (conv_state_mem[11] == load_w)|
                                (conv_state_mem[12] == load_w)|
                                (conv_state_mem[13] == load_w)|
                                (conv_state_mem[14] == load_w)|
                                (conv_state_mem[15] == load_w)|
                                (conv_state_mem[16] == load_w)|
                                (conv_state_mem[17] == load_w)|
                                (conv_state_mem[18] == load_w)|
                                (conv_state_mem[19] == load_w)|
                                (conv_state_mem[20] == load_w)|
                                (conv_state_mem[21] == load_w)|
                                (conv_state_mem[22] == load_w)|
                                (conv_state_mem[23] == load_w)|
                                (conv_state_mem[24] == load_w)|
                                (conv_state_mem[25] == load_w)|
                                (conv_state_mem[26] == load_w)|
                                (conv_state_mem[27] == load_w)|
                                (conv_state_mem[28] == load_w)|
                                (conv_state_mem[29] == load_w)|
                                (conv_state_mem[30] == load_w))?1:0;
        end
        else if (decoder_top_state_16) begin
            load_w_time = ((conv_state == load_w) |
                                (conv_state_mem[0] == load_w)|
                                (conv_state_mem[1] == load_w)|
                                (conv_state_mem[2] == load_w)|
                                (conv_state_mem[3] == load_w)|
                                (conv_state_mem[4] == load_w)|
                                (conv_state_mem[5] == load_w)|
                                (conv_state_mem[6] == load_w)|
                                (conv_state_mem[7] == load_w)|
                                (conv_state_mem[8] == load_w)|
                                (conv_state_mem[9] == load_w)|
                                (conv_state_mem[10] == load_w)|
                                (conv_state_mem[11] == load_w)|
                                (conv_state_mem[12] == load_w)|
                                (conv_state_mem[13] == load_w)|
                                (conv_state_mem[14] == load_w))?1:0;  
        end
        else if (decoder_top_state_8) begin
            load_w_time = ((conv_state == load_w) |
                                (conv_state_mem[0] == load_w)|
                                (conv_state_mem[1] == load_w)|
                                (conv_state_mem[2] == load_w)|
                                (conv_state_mem[3] == load_w)|
                                (conv_state_mem[4] == load_w)|
                                (conv_state_mem[5] == load_w)|
                                (conv_state_mem[6] == load_w))?1:0;              
        end
        else load_w_time = 0;
    end
    wire signed [W_DW-1:0] sram5_dout_lsb;
    wire signed [W_DW-1:0] sram5_dout_msb;
    wire signed [W_DW-1:0] sram4_dout_lsb;
    wire signed [W_DW-1:0] sram4_dout_msb;
    wire signed [W_DW-1:0] sram6_dout_lsb;
    wire signed [W_DW-1:0] sram6_dout_msb;
    assign sram5_dout_lsb = sram5_dout[W_DW-1:0];
    assign sram5_dout_msb = sram5_dout[2*W_DW-1:W_DW];    
    assign sram4_dout_lsb = sram4_dout[W_DW-1:0];
    assign sram4_dout_msb = sram4_dout[2*W_DW-1:W_DW];
    assign sram6_dout_lsb = sram6_dout[W_DW-1:0];
    assign sram6_dout_msb = sram6_dout[2*W_DW-1:W_DW];  

    always @(*) begin
        if (load_w_time) begin
            if (decoder_top_state == dcnn1) begin
                if (is_lsb) decoder_w = sram5_dout_lsb;
                else decoder_w = sram5_dout_msb;
            end
            else if (decoder_top_state == dcnn2) begin
                if (is_lsb) decoder_w = sram6_dout_lsb;
                else decoder_w = sram6_dout_msb;                
            end
            else begin
                if (is_lsb) decoder_w = sram4_dout_lsb;
                else decoder_w = sram4_dout_msb;                
            end
        end
        else decoder_w= 0;
    end
    reg load_b_time;
    always @(*) begin
        if (decoder_top_state_32) load_b_time = (conv_state_mem[PE_NUM-2] == load_b)? 1:0;
        else if (decoder_top_state_16) load_b_time = (conv_state_mem[PE_NUM/2-2] == load_b)? 1:0;
        else if (decoder_top_state_8) load_b_time = (conv_state_mem[PE_NUM/4-2] == load_b)? 1:0;
        else load_b_time =0;
    end
    assign sram1_en  = ((layer_top_state == load_scale) | load_b_time) ? 1:0; // scale and b are on sram 1
    assign sram4_en = (load_w_time)?( ((decoder_top_state == cnn11)|(decoder_top_state == cnn12)|(decoder_top_state == cnn21)|(decoder_top_state == cnn22))?1:0):0;
    assign sram5_en  = (load_w_time)?( (decoder_top_state == dcnn1)?1:0):0;
    assign sram6_en  = (load_w_time)?( (decoder_top_state == dcnn2)?1:0):0;

    always @(*) begin
        if (load_w_time) addr_sram = addr_decoder_w;
        else if (layer_top_state == load_scale) addr_sram = {3'B0,addr_decoder_scales_init};
        else if (load_b_time) addr_sram =  {3'B0, addr_decoder_b};
        else addr_sram = 0;
    end 


    // padding control for each pe, 01:length = 0/length_out-2, 10: length = 1/length_out-1, 11:length = 2/length_out
    reg [2:0] padding_crl;// pre: padding -1 special cases, post padding-1 special casess
    reg [2:0] padding_crl_mem[PE_NUM-2:0];
    
    reg [1:0] shift_crl;// 
    reg [2*(PE_NUM-1)-1:0] shift_crl_mem;
    assign shift_crl_all = {shift_crl_mem, shift_crl};
    

    always @(*) begin
        if (decoder_top_state == dcnn1) begin
            if (cnt_lo ==  0) begin
                padding_crl  = 1;
            end
            else if (cnt_lo == 1) begin
                padding_crl = 2;
            end
            else if (cnt_lo == 2) begin
                padding_crl = 3;
            end
            else if (cnt_lo ==  DCNN1_LENGTH_OUT -3) begin
                padding_crl = 4;
            end
            else if (cnt_lo ==  DCNN1_LENGTH_OUT -2) begin
                padding_crl = 5;
            end
            else if (cnt_lo ==  DCNN1_LENGTH_OUT -1) begin
                padding_crl = 6;
            end
            else begin
                padding_crl = 0;
            end    
        end
        else if (decoder_top_state == dcnn2) begin
            if (cnt_lo ==  0) begin
                padding_crl  = 1;
            end
            else if (cnt_lo == 1) begin
                padding_crl = 2;
            end
            else if (cnt_lo == 2) begin
                padding_crl = 3;
            end
            else if (cnt_lo ==  DCNN2_LENGTH_OUT -3) begin
                padding_crl = 4;
            end
            else if (cnt_lo ==  DCNN2_LENGTH_OUT -2) begin
                padding_crl = 5;
            end
            else if (cnt_lo ==  DCNN2_LENGTH_OUT -1) begin
                padding_crl = 6;
            end
            else begin
                padding_crl = 0;
            end    
        end 
        else if ((decoder_top_state == cnn11) |  (decoder_top_state == cnn12))  begin
            if (cnt_lo ==  0) begin
                padding_crl  = 1;
            end
            else if (cnt_lo == 1) begin
                padding_crl = 2;
            end
            else if (cnt_lo ==  CNN11_LENGTH_OUT -2) begin
                padding_crl = 3;
            end
            else if (cnt_lo ==  CNN11_LENGTH_OUT -1) begin
                padding_crl = 4;
            end
            else begin
                padding_crl = 0;
            end                
        end
        else if ((decoder_top_state == cnn21) |  (decoder_top_state == cnn22))  begin
            if (cnt_lo ==  0) begin
                padding_crl  = 1;
            end
            else if (cnt_lo == 1) begin
                padding_crl = 2;
            end
            else if (cnt_lo ==  CNN21_LENGTH_OUT -2) begin
                padding_crl = 3;
            end
            else if (cnt_lo ==  CNN21_LENGTH_OUT -1) begin
                padding_crl = 4;
            end
            else begin
                padding_crl = 0;
            end              
        end
        else  padding_crl = 0;   
    end

    reg is_odd;
    reg [PE_NUM-2:0] is_odd_mem;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) is_odd <= 0;
        else begin
            if (decoder_top_state_dcnn) begin
                if (conv_state == load_a) begin
                    is_odd <= ~is_odd;
                end
                else begin
                    is_odd <= is_odd;
                end
                end
            else begin
                is_odd<= 0;
            end
        end
    end

    always @(*) begin
        if (decoder_top_state_dcnn) begin
            if (conv_state == load_a) begin
                if ((padding_crl !=  0) & (padding_crl != 6)) begin
                    shift_crl  = 0;
                end
                else if (padding_crl == 6) begin
                    shift_crl = 2; //reset
                end
                else if (~is_odd) begin
                    shift_crl = 1; // shift for one 
                end
                else begin
                    shift_crl = 0;
                end
            end
            else begin
                shift_crl = 0;
            end                 
        end
        else begin
            if (conv_state == load_a) begin
                if ((padding_crl !=  0) & (padding_crl != 4)) begin
                    shift_crl  = 0;
                end
                else if (padding_crl == 4) begin
                    shift_crl = 2; //reset
                end
                else  begin
                    shift_crl = 1; // shift for one 
                end
            end
            else begin
                shift_crl = 0;
            end
        end
    
    end

    integer r;
    integer t;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin 
            shift_crl_mem <= 0;
            is_odd_mem <= 0;
            for (r = 0; r < PE_NUM-1; r = r+1) begin
                padding_crl_mem[r] <= 0;
            end
        end
        else begin
            padding_crl_mem[0] <= padding_crl;
            shift_crl_mem [1:0] <= shift_crl;
            is_odd_mem[0] <= is_odd;
            for (t = 1; t < PE_NUM-1; t = t+1) begin
                padding_crl_mem[t] <= padding_crl_mem[t-1];
                shift_crl_mem [(t+1)*2-1-:2] <= shift_crl_mem[t*2-1-:2];
                is_odd_mem[t] <= is_odd_mem[t-1];
            end
        end
    end


/////////////////////////// SRAM SPAD IN OUT ///////////////////////////////////////////
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_1;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_2;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_3;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_4;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_5;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_6;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_7;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_8;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_9;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_10;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_11;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_12;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_13;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_14;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_15;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_16;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_17;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_18;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_19;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_20;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_21;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_22;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_23;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_24;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_25;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_26;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_27;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_28;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_29;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_30;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_31;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_for_32;
    reg [$clog2(SPAD_DEPTH+1)-1:0] cnt_re_sram;
    reg  addr_sram_act_re_end_1;
    reg  addr_sram_act_re_end_2;
    reg  addr_sram_act_re_end_3;
    reg  addr_sram_act_re_end_4;
    reg  addr_sram_act_re_end_5;
    reg  addr_sram_act_re_end_6;
    reg  addr_sram_act_re_end_7;
    reg  addr_sram_act_re_end_8;
    reg  addr_sram_act_re_end_9;
    reg  addr_sram_act_re_end_10;
    reg  addr_sram_act_re_end_11;
    reg  addr_sram_act_re_end_12;
    reg  addr_sram_act_re_end_13;
    reg  addr_sram_act_re_end_14;
    reg  addr_sram_act_re_end_15;
    reg  addr_sram_act_re_end_16;
    reg  addr_sram_act_re_end_17;
    reg  addr_sram_act_re_end_18;
    reg  addr_sram_act_re_end_19;
    reg  addr_sram_act_re_end_20;
    reg  addr_sram_act_re_end_21;
    reg  addr_sram_act_re_end_22;
    reg  addr_sram_act_re_end_23;
    reg  addr_sram_act_re_end_24;
    reg  addr_sram_act_re_end_25;
    reg  addr_sram_act_re_end_26;
    reg  addr_sram_act_re_end_27;
    reg  addr_sram_act_re_end_28;
    reg  addr_sram_act_re_end_29;
    reg  addr_sram_act_re_end_30;
    reg  addr_sram_act_re_end_31;
    reg  addr_sram_act_re_end_32;
    reg [SRAM8192_AW -1 : 0] addr_sram_act_re; //zong 
    

    


    reg [SRAM8192_AW -1 : 0] addr_sram_act_we;
    reg addr_sram_act_we_end;
    reg [$clog2(2+1)-1:0] cnt_we_sram;

    localparam ADDR_DCNN1_SRAM_ACT_INIT = 2048; //2048-6143
    localparam ADDR_CNN11_SRAM_ACT_INIT_0 = 6144;//6144-8191,0-2047
    localparam ADDR_CNN11_SRAM_ACT_INIT_1 = 0;//
    localparam ADDR_CNN12_SRAM_ACT_INIT = 2048;
    localparam ADDR_DCNN2_SRAM_ACT_INIT = 4096;
    localparam ADDR_CNN21_SRAM_ACT_INIT = 6144;

    // assign spad_a_we_en_1_32 = {spad_a_we_en_2_32,spad1_a_we_en};

    always @(negedge sclk or negedge rst_n) begin
        if (!rst_n) begin
            addr_sram_act_re_end_1 <= 0;
            addr_sram_act_re_end_2 <= 0;
            addr_sram_act_re_end_3 <= 0;
            addr_sram_act_re_end_4 <= 0;
            addr_sram_act_re_end_5 <= 0;
            addr_sram_act_re_end_6 <= 0;
            addr_sram_act_re_end_7 <= 0;
            addr_sram_act_re_end_8 <= 0;
            addr_sram_act_re_end_9 <= 0;
            addr_sram_act_re_end_10 <= 0;
            addr_sram_act_re_end_11 <= 0;
            addr_sram_act_re_end_12 <= 0;
            addr_sram_act_re_end_13 <= 0;
            addr_sram_act_re_end_14 <= 0;
            addr_sram_act_re_end_15 <= 0;
            addr_sram_act_re_end_16 <= 0;
            addr_sram_act_re_end_17 <= 0;
            addr_sram_act_re_end_18<= 0;
            addr_sram_act_re_end_19 <= 0;
            addr_sram_act_re_end_20 <= 0;
            addr_sram_act_re_end_21 <= 0;
            addr_sram_act_re_end_22 <= 0;
            addr_sram_act_re_end_23 <= 0;
            addr_sram_act_re_end_24 <= 0;
            addr_sram_act_re_end_25 <= 0;
            addr_sram_act_re_end_26 <= 0;
            addr_sram_act_re_end_27 <= 0;
            addr_sram_act_re_end_28 <= 0;
            addr_sram_act_re_end_29 <= 0;
            addr_sram_act_re_end_30 <= 0;
            addr_sram_act_re_end_31 <= 0;
            addr_sram_act_re_end_32 <= 0;
            addr_sram_act_for_1 <= ADDR_LSTM10_SRAM_ACT_INIT;
            addr_sram_act_for_2 <= ADDR_LSTM10_SRAM_ACT_INIT + 1;
            addr_sram_act_for_3 <= ADDR_LSTM10_SRAM_ACT_INIT + 2;
            addr_sram_act_for_4 <= ADDR_LSTM10_SRAM_ACT_INIT + 3;
            addr_sram_act_for_5 <= ADDR_LSTM10_SRAM_ACT_INIT + 4;
            addr_sram_act_for_6 <= ADDR_LSTM10_SRAM_ACT_INIT + 5;
            addr_sram_act_for_7 <= ADDR_LSTM10_SRAM_ACT_INIT + 6;
            addr_sram_act_for_8 <= ADDR_LSTM10_SRAM_ACT_INIT + 7;
            addr_sram_act_for_9 <= ADDR_LSTM10_SRAM_ACT_INIT + 8;
            addr_sram_act_for_10 <= ADDR_LSTM10_SRAM_ACT_INIT + 9;
            addr_sram_act_for_11 <= ADDR_LSTM10_SRAM_ACT_INIT + 10;
            addr_sram_act_for_12 <= ADDR_LSTM10_SRAM_ACT_INIT + 11;
            addr_sram_act_for_13 <= ADDR_LSTM10_SRAM_ACT_INIT + 12;
            addr_sram_act_for_14 <= ADDR_LSTM10_SRAM_ACT_INIT + 13;
            addr_sram_act_for_15 <= ADDR_LSTM10_SRAM_ACT_INIT + 14;
            addr_sram_act_for_16 <= ADDR_LSTM10_SRAM_ACT_INIT + 15;
            addr_sram_act_for_17 <= ADDR_LSTM10_SRAM_ACT_INIT + 16;
            addr_sram_act_for_18 <= ADDR_LSTM10_SRAM_ACT_INIT + 17;
            addr_sram_act_for_19 <= ADDR_LSTM10_SRAM_ACT_INIT + 18;
            addr_sram_act_for_20 <= ADDR_LSTM10_SRAM_ACT_INIT + 19;
            addr_sram_act_for_21 <= ADDR_LSTM10_SRAM_ACT_INIT + 20;
            addr_sram_act_for_22 <= ADDR_LSTM10_SRAM_ACT_INIT + 21;
            addr_sram_act_for_23 <= ADDR_LSTM10_SRAM_ACT_INIT + 22;
            addr_sram_act_for_24 <= ADDR_LSTM10_SRAM_ACT_INIT + 23;
            addr_sram_act_for_25 <= ADDR_LSTM10_SRAM_ACT_INIT + 24;
            addr_sram_act_for_26 <= ADDR_LSTM10_SRAM_ACT_INIT + 25;
            addr_sram_act_for_27 <= ADDR_LSTM10_SRAM_ACT_INIT + 26;
            addr_sram_act_for_28 <= ADDR_LSTM10_SRAM_ACT_INIT + 27;
            addr_sram_act_for_29 <= ADDR_LSTM10_SRAM_ACT_INIT + 28;
            addr_sram_act_for_30 <= ADDR_LSTM10_SRAM_ACT_INIT + 29;
            addr_sram_act_for_31 <= ADDR_LSTM10_SRAM_ACT_INIT + 30;
            addr_sram_act_for_32 <= ADDR_LSTM10_SRAM_ACT_INIT + 31;
                        // addr_sram_act_for_1 <= ADDR_DCNN2_SRAM_ACT_INIT;
                        // addr_sram_act_for_2 <= ADDR_DCNN2_SRAM_ACT_INIT + CNN21_LENGTH_IN;
                        // addr_sram_act_for_3 <= ADDR_DCNN2_SRAM_ACT_INIT + 2 * CNN21_LENGTH_IN;
                        // addr_sram_act_for_4 <= ADDR_DCNN2_SRAM_ACT_INIT + 3 * CNN21_LENGTH_IN;
                        // addr_sram_act_for_5 <= ADDR_DCNN2_SRAM_ACT_INIT + 4 * CNN21_LENGTH_IN;
                        // addr_sram_act_for_6 <= ADDR_DCNN2_SRAM_ACT_INIT + 5 * CNN21_LENGTH_IN;
                        // addr_sram_act_for_7 <= ADDR_DCNN2_SRAM_ACT_INIT + 6 * CNN21_LENGTH_IN;
                        // addr_sram_act_for_8 <= ADDR_DCNN2_SRAM_ACT_INIT + 7 * CNN21_LENGTH_IN;
                        // addr_sram_act_for_9 <= ADDR_DCNN2_SRAM_ACT_INIT + 8 * CNN21_LENGTH_IN;
                        // addr_sram_act_for_10 <= ADDR_DCNN2_SRAM_ACT_INIT + 9 * CNN21_LENGTH_IN;
                        // addr_sram_act_for_11 <= ADDR_DCNN2_SRAM_ACT_INIT + 10 * CNN21_LENGTH_IN;
                        // addr_sram_act_for_12 <= ADDR_DCNN2_SRAM_ACT_INIT + 11 * CNN21_LENGTH_IN;
                        // addr_sram_act_for_13 <= ADDR_DCNN2_SRAM_ACT_INIT + 12 * CNN21_LENGTH_IN;
                        // addr_sram_act_for_14 <= ADDR_DCNN2_SRAM_ACT_INIT + 13 * CNN21_LENGTH_IN;
                        // addr_sram_act_for_15 <= ADDR_DCNN2_SRAM_ACT_INIT + 14 * CNN21_LENGTH_IN;
                        // addr_sram_act_for_16 <= ADDR_DCNN2_SRAM_ACT_INIT + 15 * CNN21_LENGTH_IN;
                        // addr_sram_act_for_17 <= 0 ;
                        // addr_sram_act_for_18 <= 0;
                        // addr_sram_act_for_19 <= 0;
                        // addr_sram_act_for_20 <= 0;
                        // addr_sram_act_for_21 <= 0;
                        // addr_sram_act_for_22 <= 0;
                        // addr_sram_act_for_23 <= 0;
                        // addr_sram_act_for_24 <= 0;
                        // addr_sram_act_for_25 <= 0;
                        // addr_sram_act_for_26 <= 0;
                        // addr_sram_act_for_27 <= 0;
                        // addr_sram_act_for_28 <= 0;
                        // addr_sram_act_for_29 <= 0;
                        // addr_sram_act_for_30 <= 0;
                        // addr_sram_act_for_31 <= 0;
                        // addr_sram_act_for_32 <= 0;   
            cnt_re_sram  <= 0;            
        end
        else begin
            if (decoder_top_state == dcnn1) begin
                if (sram_act_we) begin
                    cnt_re_sram  <= cnt_re_sram;  
                end
                else begin
                    if ((conv_state == load_a) & !addr_sram_act_re_end_1) begin                    
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_1 <= 0;
                            case (padding_crl)
                                0: begin
                                    if (~is_odd) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) 
                                            addr_sram_act_for_1 <= (cnt_bt)? addr_sram_act_for_1 - DCNN1_CHIN/2 : addr_sram_act_for_1 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_1 <= addr_sram_act_for_1;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) 
                                            addr_sram_act_for_1 <= (cnt_bt)? addr_sram_act_for_1 - DCNN1_CHIN/2 : addr_sram_act_for_1 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_1 <= addr_sram_act_for_1;                                    
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_1 <= (cnt_bt)? addr_sram_act_for_1 - DCNN1_CHIN/2:addr_sram_act_for_1 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_1 <= addr_sram_act_for_1;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_1 <= (cnt_bt)? addr_sram_act_for_1 - DCNN1_CHIN/2: addr_sram_act_for_1 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_1 <= (cnt_bt)?  addr_sram_act_for_1 - DCNN1_CHIN/2: addr_sram_act_for_1 + DCNN1_CHIN/2;
                                    end  
                                    else begin
                                        addr_sram_act_for_1 <= addr_sram_act_for_1;
                                    end                              
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_1 <= (cnt_bt)?  addr_sram_act_for_1 - DCNN1_CHIN/2: addr_sram_act_for_1 + DCNN1_CHIN/2;
                                    end  
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_1 <= (cnt_bt)?  addr_sram_act_for_1 - DCNN1_CHIN/2: addr_sram_act_for_1 + DCNN1_CHIN/2;
                                    end  
                                    else begin
                                        addr_sram_act_for_1 <= addr_sram_act_for_1;
                                    end                                  
                                end
                                4:begin
                                    if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_1 <= (cnt_bt)? addr_sram_act_for_1 - DCNN1_CHIN/2: addr_sram_act_for_1 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_1 <= addr_sram_act_for_1;
                                    end                                
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_1 <= (cnt_bt)? addr_sram_act_for_1 - DCNN1_CHIN/2: addr_sram_act_for_1 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_1 <= (cnt_bt)? addr_sram_act_for_1 - DCNN1_CHIN/2: addr_sram_act_for_1 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_1 <= addr_sram_act_for_1;
                                    end                             
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_1 <= (cnt_bt)? addr_sram_act_for_1 - DCNN1_CHIN/2:addr_sram_act_for_1 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_1 <= (cnt_bt)? addr_sram_act_for_1 - DCNN1_CHIN/2:addr_sram_act_for_1 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_1 <= addr_sram_act_for_1;
                                    end                                   
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_1 <= 1;
                            if (cnt_bt == 0) begin
                                if (padding_crl == 0) addr_sram_act_for_1 <= (is_odd)?  addr_sram_act_for_1 - 3*DCNN1_CHIN/2:  addr_sram_act_for_1 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl == 1)  | (padding_crl == 2) | (padding_crl == 3)) addr_sram_act_for_1 <= ADDR_LSTM10_SRAM_ACT_INIT;
                                else if ((padding_crl == 4)  | (padding_crl == 5) ) addr_sram_act_for_1 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2;
                                else addr_sram_act_for_1 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl == 0) addr_sram_act_for_1 <= (is_odd)? addr_sram_act_for_1 + 3*DCNN1_CHIN/2 : addr_sram_act_for_1  + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl == 1)  | (padding_crl == 2) | (padding_crl == 3)) addr_sram_act_for_1 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 ;
                                else if ((padding_crl == 4)  | (padding_crl == 5) ) addr_sram_act_for_1 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 ;                            
                                else addr_sram_act_for_1 <= ADDR_LSTM10_SRAM_ACT_INIT;//for bt = 0
                            end
                        end
                    end
                    else if ((conv_state_mem[0] == load_a) & !addr_sram_act_re_end_2) begin
                        addr_sram_act_re_end_1 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_2 <= 0;
                            case (padding_crl_mem[0])
                                0: begin
                                    if (~is_odd_mem[0]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_2 <= (cnt_bt_mem[0])? addr_sram_act_for_2 - DCNN1_CHIN/2 : addr_sram_act_for_2 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_2 <= addr_sram_act_for_2;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_2 <= (cnt_bt_mem[0])? addr_sram_act_for_2 - DCNN1_CHIN/2 : addr_sram_act_for_2 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_2 <= addr_sram_act_for_2;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_2 <= (cnt_bt_mem[0])? addr_sram_act_for_2 - DCNN1_CHIN/2:addr_sram_act_for_2 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_2 <= addr_sram_act_for_2;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_2 <= (cnt_bt_mem[0])? addr_sram_act_for_2 - DCNN1_CHIN/2: addr_sram_act_for_2 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_2 <= (cnt_bt_mem[0])?  addr_sram_act_for_2 - DCNN1_CHIN/2: addr_sram_act_for_2 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_2 <= addr_sram_act_for_2;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_2 <= (cnt_bt_mem[0])?  addr_sram_act_for_2 - DCNN1_CHIN/2: addr_sram_act_for_2 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_2 <= (cnt_bt_mem[0])?  addr_sram_act_for_2 - DCNN1_CHIN/2: addr_sram_act_for_2 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_2 <= addr_sram_act_for_2;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_2 <= (cnt_bt_mem[0])? addr_sram_act_for_2 - DCNN1_CHIN/2: addr_sram_act_for_2 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_2 <= addr_sram_act_for_2;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_2 <= (cnt_bt_mem[0])? addr_sram_act_for_2 - DCNN1_CHIN/2: addr_sram_act_for_2 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_2 <= (cnt_bt_mem[0])? addr_sram_act_for_2 - DCNN1_CHIN/2: addr_sram_act_for_2 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_2 <= addr_sram_act_for_2;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_2 <= (cnt_bt_mem[0])? addr_sram_act_for_2 - DCNN1_CHIN/2:addr_sram_act_for_2 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_2 <= (cnt_bt_mem[0])? addr_sram_act_for_2 - DCNN1_CHIN/2:addr_sram_act_for_2 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_2 <= addr_sram_act_for_2;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_2 <= 1;
                            if (cnt_bt_mem[0] == 0) begin
                                if (padding_crl_mem[0] == 0) addr_sram_act_for_2 <= (is_odd_mem[0])?  addr_sram_act_for_2 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_2 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[0] == 1)  | (padding_crl_mem[0] == 2) | (padding_crl_mem[0] == 3)) addr_sram_act_for_2 <= ADDR_LSTM10_SRAM_ACT_INIT + 1;
                                else if ((padding_crl_mem[0] == 4)  | (padding_crl_mem[0] == 5) ) addr_sram_act_for_2 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 1;
                                else addr_sram_act_for_2 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 1 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[0] == 0) addr_sram_act_for_2 <= (is_odd_mem[0])? addr_sram_act_for_2 + 3*DCNN1_CHIN/2 : addr_sram_act_for_2 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[0] == 1)  | (padding_crl_mem[0] == 2) | (padding_crl_mem[0] == 3)) addr_sram_act_for_2 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 1;
                                else if ((padding_crl_mem[0] == 4)  | (padding_crl_mem[0] == 5)) addr_sram_act_for_2 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 1;
                                else addr_sram_act_for_2 <= ADDR_LSTM10_SRAM_ACT_INIT + 1;
                            end
                        end
                    end
                    else if ((conv_state_mem[1] == load_a) & !addr_sram_act_re_end_3) begin
                        addr_sram_act_re_end_2 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_3 <= 0;
                            case (padding_crl_mem[1])
                                0: begin
                                    if (~is_odd_mem[1]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_3 <= (cnt_bt_mem[1])? addr_sram_act_for_3 - DCNN1_CHIN/2 : addr_sram_act_for_3 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_3 <= addr_sram_act_for_3;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_3 <= (cnt_bt_mem[1])? addr_sram_act_for_3 - DCNN1_CHIN/2 : addr_sram_act_for_3 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_3 <= addr_sram_act_for_3;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_3 <= (cnt_bt_mem[1])? addr_sram_act_for_3 - DCNN1_CHIN/2:addr_sram_act_for_3 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_3 <= addr_sram_act_for_3;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_3 <= (cnt_bt_mem[1])? addr_sram_act_for_3 - DCNN1_CHIN/2: addr_sram_act_for_3 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_3 <= (cnt_bt_mem[1])?  addr_sram_act_for_3 - DCNN1_CHIN/2: addr_sram_act_for_3 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_3 <= addr_sram_act_for_3;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_3 <= (cnt_bt_mem[1])?  addr_sram_act_for_3 - DCNN1_CHIN/2: addr_sram_act_for_3 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_3 <= (cnt_bt_mem[1])?  addr_sram_act_for_3 - DCNN1_CHIN/2: addr_sram_act_for_3 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_3 <= addr_sram_act_for_3;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_3 <= (cnt_bt_mem[1])? addr_sram_act_for_3 - DCNN1_CHIN/2: addr_sram_act_for_3 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_3 <= addr_sram_act_for_3;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_3 <= (cnt_bt_mem[1])? addr_sram_act_for_3 - DCNN1_CHIN/2: addr_sram_act_for_3 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_3 <= (cnt_bt_mem[1])? addr_sram_act_for_3 - DCNN1_CHIN/2: addr_sram_act_for_3 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_3 <= addr_sram_act_for_3;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_3 <= (cnt_bt_mem[1])? addr_sram_act_for_3 - DCNN1_CHIN/2:addr_sram_act_for_3 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_3 <= (cnt_bt_mem[1])? addr_sram_act_for_3 - DCNN1_CHIN/2:addr_sram_act_for_3 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_3 <= addr_sram_act_for_3;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_3 <= 1;
                            if (cnt_bt_mem[1] == 0) begin
                                if (padding_crl_mem[1] == 0) addr_sram_act_for_3 <= (is_odd_mem[1])?  addr_sram_act_for_3 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_3 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[1] == 1)  | (padding_crl_mem[1] == 2) | (padding_crl_mem[1] == 3)) addr_sram_act_for_3 <= ADDR_LSTM10_SRAM_ACT_INIT + 2;
                                else if ((padding_crl_mem[1] == 4)  | (padding_crl_mem[1] == 5) ) addr_sram_act_for_3 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 2;
                                else addr_sram_act_for_3 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 2 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[1] == 0) addr_sram_act_for_3 <= (is_odd_mem[1])? addr_sram_act_for_3 + 3*DCNN1_CHIN/2 : addr_sram_act_for_3 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[1] == 1)  | (padding_crl_mem[1] == 2) | (padding_crl_mem[1] == 3)) addr_sram_act_for_3 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 2;
                                else if ((padding_crl_mem[1] == 4)  | (padding_crl_mem[1] == 5)) addr_sram_act_for_3 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 2;
                                else addr_sram_act_for_3 <= ADDR_LSTM10_SRAM_ACT_INIT + 2;
                            end
                        end
                    end
                    else if ((conv_state_mem[2] == load_a) & !addr_sram_act_re_end_4) begin
                        addr_sram_act_re_end_3 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_4 <= 0;
                            case (padding_crl_mem[2])
                                0: begin
                                    if (~is_odd_mem[2]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_4 <= (cnt_bt_mem[2])? addr_sram_act_for_4 - DCNN1_CHIN/2 : addr_sram_act_for_4 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_4 <= addr_sram_act_for_4;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_4 <= (cnt_bt_mem[2])? addr_sram_act_for_4 - DCNN1_CHIN/2 : addr_sram_act_for_4 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_4 <= addr_sram_act_for_4;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_4 <= (cnt_bt_mem[2])? addr_sram_act_for_4 - DCNN1_CHIN/2:addr_sram_act_for_4 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_4 <= addr_sram_act_for_4;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_4 <= (cnt_bt_mem[2])? addr_sram_act_for_4 - DCNN1_CHIN/2: addr_sram_act_for_4 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_4 <= (cnt_bt_mem[2])?  addr_sram_act_for_4 - DCNN1_CHIN/2: addr_sram_act_for_4 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_4 <= addr_sram_act_for_4;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_4 <= (cnt_bt_mem[2])?  addr_sram_act_for_4 - DCNN1_CHIN/2: addr_sram_act_for_4 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_4 <= (cnt_bt_mem[2])?  addr_sram_act_for_4 - DCNN1_CHIN/2: addr_sram_act_for_4 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_4 <= addr_sram_act_for_4;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_4 <= (cnt_bt_mem[2])? addr_sram_act_for_4 - DCNN1_CHIN/2: addr_sram_act_for_4 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_4 <= addr_sram_act_for_4;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_4 <= (cnt_bt_mem[2])? addr_sram_act_for_4 - DCNN1_CHIN/2: addr_sram_act_for_4 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_4 <= (cnt_bt_mem[2])? addr_sram_act_for_4 - DCNN1_CHIN/2: addr_sram_act_for_4 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_4 <= addr_sram_act_for_4;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_4 <= (cnt_bt_mem[2])? addr_sram_act_for_4 - DCNN1_CHIN/2:addr_sram_act_for_4 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_4 <= (cnt_bt_mem[2])? addr_sram_act_for_4 - DCNN1_CHIN/2:addr_sram_act_for_4 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_4 <= addr_sram_act_for_4;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_4 <= 1;
                            if (cnt_bt_mem[2] == 0) begin
                                if (padding_crl_mem[2] == 0) addr_sram_act_for_4 <= (is_odd_mem[2])?  addr_sram_act_for_4 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_4 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[2] == 1)  | (padding_crl_mem[2] == 2) | (padding_crl_mem[2] == 3)) addr_sram_act_for_4 <= ADDR_LSTM10_SRAM_ACT_INIT + 3;
                                else if ((padding_crl_mem[2] == 4)  | (padding_crl_mem[2] == 5) ) addr_sram_act_for_4 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 3;
                                else addr_sram_act_for_4 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 3 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[2] == 0) addr_sram_act_for_4 <= (is_odd_mem[2])? addr_sram_act_for_4 + 3*DCNN1_CHIN/2 : addr_sram_act_for_4 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[2] == 1)  | (padding_crl_mem[2] == 2) | (padding_crl_mem[2] == 3)) addr_sram_act_for_4 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 3;
                                else if ((padding_crl_mem[2] == 4)  | (padding_crl_mem[2] == 5)) addr_sram_act_for_4 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 3;
                                else addr_sram_act_for_4 <= ADDR_LSTM10_SRAM_ACT_INIT + 3;
                            end
                        end
                    end
                    else if ((conv_state_mem[3] == load_a) & !addr_sram_act_re_end_5) begin
                        addr_sram_act_re_end_4 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_5 <= 0;
                            case (padding_crl_mem[3])
                                0: begin
                                    if (~is_odd_mem[3]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_5 <= (cnt_bt_mem[3])? addr_sram_act_for_5 - DCNN1_CHIN/2 : addr_sram_act_for_5 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_5 <= addr_sram_act_for_5;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_5 <= (cnt_bt_mem[3])? addr_sram_act_for_5 - DCNN1_CHIN/2 : addr_sram_act_for_5 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_5 <= addr_sram_act_for_5;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_5 <= (cnt_bt_mem[3])? addr_sram_act_for_5 - DCNN1_CHIN/2:addr_sram_act_for_5 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_5 <= addr_sram_act_for_5;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_5 <= (cnt_bt_mem[3])? addr_sram_act_for_5 - DCNN1_CHIN/2: addr_sram_act_for_5 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_5 <= (cnt_bt_mem[3])?  addr_sram_act_for_5 - DCNN1_CHIN/2: addr_sram_act_for_5 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_5 <= addr_sram_act_for_5;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_5 <= (cnt_bt_mem[3])?  addr_sram_act_for_5 - DCNN1_CHIN/2: addr_sram_act_for_5 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_5 <= (cnt_bt_mem[3])?  addr_sram_act_for_5 - DCNN1_CHIN/2: addr_sram_act_for_5 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_5 <= addr_sram_act_for_5;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_5 <= (cnt_bt_mem[3])? addr_sram_act_for_5 - DCNN1_CHIN/2: addr_sram_act_for_5 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_5 <= addr_sram_act_for_5;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_5 <= (cnt_bt_mem[3])? addr_sram_act_for_5 - DCNN1_CHIN/2: addr_sram_act_for_5 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_5 <= (cnt_bt_mem[3])? addr_sram_act_for_5 - DCNN1_CHIN/2: addr_sram_act_for_5 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_5 <= addr_sram_act_for_5;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_5 <= (cnt_bt_mem[3])? addr_sram_act_for_5 - DCNN1_CHIN/2:addr_sram_act_for_5 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_5 <= (cnt_bt_mem[3])? addr_sram_act_for_5 - DCNN1_CHIN/2:addr_sram_act_for_5 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_5 <= addr_sram_act_for_5;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_5 <= 1;
                            if (cnt_bt_mem[3] == 0) begin
                                if (padding_crl_mem[3] == 0) addr_sram_act_for_5 <= (is_odd_mem[3])?  addr_sram_act_for_5 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_5 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[3] == 1)  | (padding_crl_mem[3] == 2) | (padding_crl_mem[3] == 3)) addr_sram_act_for_5 <= ADDR_LSTM10_SRAM_ACT_INIT + 4;
                                else if ((padding_crl_mem[3] == 4)  | (padding_crl_mem[3] == 5) ) addr_sram_act_for_5 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 4;
                                else addr_sram_act_for_5 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 4 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[3] == 0) addr_sram_act_for_5 <= (is_odd_mem[3])? addr_sram_act_for_5 + 3*DCNN1_CHIN/2 : addr_sram_act_for_5 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[3] == 1)  | (padding_crl_mem[3] == 2) | (padding_crl_mem[3] == 3)) addr_sram_act_for_5 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 4;
                                else if ((padding_crl_mem[3] == 4)  | (padding_crl_mem[3] == 5)) addr_sram_act_for_5 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 4;
                                else addr_sram_act_for_5 <= ADDR_LSTM10_SRAM_ACT_INIT + 4;
                            end
                        end
                    end
                    else if ((conv_state_mem[4] == load_a) & !addr_sram_act_re_end_6) begin
                        addr_sram_act_re_end_5 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_6 <= 0;
                            case (padding_crl_mem[4])
                                0: begin
                                    if (~is_odd_mem[4]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_6 <= (cnt_bt_mem[4])? addr_sram_act_for_6 - DCNN1_CHIN/2 : addr_sram_act_for_6 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_6 <= addr_sram_act_for_6;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_6 <= (cnt_bt_mem[4])? addr_sram_act_for_6 - DCNN1_CHIN/2 : addr_sram_act_for_6 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_6 <= addr_sram_act_for_6;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_6 <= (cnt_bt_mem[4])? addr_sram_act_for_6 - DCNN1_CHIN/2:addr_sram_act_for_6 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_6 <= addr_sram_act_for_6;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_6 <= (cnt_bt_mem[4])? addr_sram_act_for_6 - DCNN1_CHIN/2: addr_sram_act_for_6 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_6 <= (cnt_bt_mem[4])?  addr_sram_act_for_6 - DCNN1_CHIN/2: addr_sram_act_for_6 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_6 <= addr_sram_act_for_6;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_6 <= (cnt_bt_mem[4])?  addr_sram_act_for_6 - DCNN1_CHIN/2: addr_sram_act_for_6 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_6 <= (cnt_bt_mem[4])?  addr_sram_act_for_6 - DCNN1_CHIN/2: addr_sram_act_for_6 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_6 <= addr_sram_act_for_6;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_6 <= (cnt_bt_mem[4])? addr_sram_act_for_6 - DCNN1_CHIN/2: addr_sram_act_for_6 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_6 <= addr_sram_act_for_6;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_6 <= (cnt_bt_mem[4])? addr_sram_act_for_6 - DCNN1_CHIN/2: addr_sram_act_for_6 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_6 <= (cnt_bt_mem[4])? addr_sram_act_for_6 - DCNN1_CHIN/2: addr_sram_act_for_6 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_6 <= addr_sram_act_for_6;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_6 <= (cnt_bt_mem[4])? addr_sram_act_for_6 - DCNN1_CHIN/2:addr_sram_act_for_6 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_6 <= (cnt_bt_mem[4])? addr_sram_act_for_6 - DCNN1_CHIN/2:addr_sram_act_for_6 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_6 <= addr_sram_act_for_6;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_6 <= 1;
                            if (cnt_bt_mem[4] == 0) begin
                                if (padding_crl_mem[4] == 0) addr_sram_act_for_6 <= (is_odd_mem[4])?  addr_sram_act_for_6 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_6 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[4] == 1)  | (padding_crl_mem[4] == 2) | (padding_crl_mem[4] == 3)) addr_sram_act_for_6 <= ADDR_LSTM10_SRAM_ACT_INIT + 5;
                                else if ((padding_crl_mem[4] == 4)  | (padding_crl_mem[4] == 5) ) addr_sram_act_for_6 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 5;
                                else addr_sram_act_for_6 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 5 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[4] == 0) addr_sram_act_for_6 <= (is_odd_mem[4])? addr_sram_act_for_6 + 3*DCNN1_CHIN/2 : addr_sram_act_for_6 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[4] == 1)  | (padding_crl_mem[4] == 2) | (padding_crl_mem[4] == 3)) addr_sram_act_for_6 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 5;
                                else if ((padding_crl_mem[4] == 4)  | (padding_crl_mem[4] == 5)) addr_sram_act_for_6 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 5;
                                else addr_sram_act_for_6 <= ADDR_LSTM10_SRAM_ACT_INIT + 5;
                            end
                        end
                    end
                    else if ((conv_state_mem[5] == load_a) & !addr_sram_act_re_end_7) begin
                        addr_sram_act_re_end_6 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_7 <= 0;
                            case (padding_crl_mem[5])
                                0: begin
                                    if (~is_odd_mem[5]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_7 <= (cnt_bt_mem[5])? addr_sram_act_for_7 - DCNN1_CHIN/2 : addr_sram_act_for_7 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_7 <= addr_sram_act_for_7;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_7 <= (cnt_bt_mem[5])? addr_sram_act_for_7 - DCNN1_CHIN/2 : addr_sram_act_for_7 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_7 <= addr_sram_act_for_7;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_7 <= (cnt_bt_mem[5])? addr_sram_act_for_7 - DCNN1_CHIN/2:addr_sram_act_for_7 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_7 <= addr_sram_act_for_7;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_7 <= (cnt_bt_mem[5])? addr_sram_act_for_7 - DCNN1_CHIN/2: addr_sram_act_for_7 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_7 <= (cnt_bt_mem[5])?  addr_sram_act_for_7 - DCNN1_CHIN/2: addr_sram_act_for_7 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_7 <= addr_sram_act_for_7;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_7 <= (cnt_bt_mem[5])?  addr_sram_act_for_7 - DCNN1_CHIN/2: addr_sram_act_for_7 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_7 <= (cnt_bt_mem[5])?  addr_sram_act_for_7 - DCNN1_CHIN/2: addr_sram_act_for_7 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_7 <= addr_sram_act_for_7;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_7 <= (cnt_bt_mem[5])? addr_sram_act_for_7 - DCNN1_CHIN/2: addr_sram_act_for_7 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_7 <= addr_sram_act_for_7;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_7 <= (cnt_bt_mem[5])? addr_sram_act_for_7 - DCNN1_CHIN/2: addr_sram_act_for_7 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_7 <= (cnt_bt_mem[5])? addr_sram_act_for_7 - DCNN1_CHIN/2: addr_sram_act_for_7 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_7 <= addr_sram_act_for_7;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_7 <= (cnt_bt_mem[5])? addr_sram_act_for_7 - DCNN1_CHIN/2:addr_sram_act_for_7 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_7 <= (cnt_bt_mem[5])? addr_sram_act_for_7 - DCNN1_CHIN/2:addr_sram_act_for_7 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_7 <= addr_sram_act_for_7;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_7 <= 1;
                            if (cnt_bt_mem[5] == 0) begin
                                if (padding_crl_mem[5] == 0) addr_sram_act_for_7 <= (is_odd_mem[5])?  addr_sram_act_for_7 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_7 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[5] == 1)  | (padding_crl_mem[5] == 2) | (padding_crl_mem[5] == 3)) addr_sram_act_for_7 <= ADDR_LSTM10_SRAM_ACT_INIT + 6;
                                else if ((padding_crl_mem[5] == 4)  | (padding_crl_mem[5] == 5) ) addr_sram_act_for_7 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 6;
                                else addr_sram_act_for_7 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 6 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[5] == 0) addr_sram_act_for_7 <= (is_odd_mem[5])? addr_sram_act_for_7 + 3*DCNN1_CHIN/2 : addr_sram_act_for_7 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[5] == 1)  | (padding_crl_mem[5] == 2) | (padding_crl_mem[5] == 3)) addr_sram_act_for_7 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 6;
                                else if ((padding_crl_mem[5] == 4)  | (padding_crl_mem[5] == 5)) addr_sram_act_for_7 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 6;
                                else addr_sram_act_for_7 <= ADDR_LSTM10_SRAM_ACT_INIT + 6;
                            end
                        end
                    end
                    else if ((conv_state_mem[6] == load_a) & !addr_sram_act_re_end_8) begin
                        addr_sram_act_re_end_7 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_8 <= 0;
                            case (padding_crl_mem[6])
                                0: begin
                                    if (~is_odd_mem[6]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_8 <= (cnt_bt_mem[6])? addr_sram_act_for_8 - DCNN1_CHIN/2 : addr_sram_act_for_8 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_8 <= addr_sram_act_for_8;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_8 <= (cnt_bt_mem[6])? addr_sram_act_for_8 - DCNN1_CHIN/2 : addr_sram_act_for_8 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_8 <= addr_sram_act_for_8;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_8 <= (cnt_bt_mem[6])? addr_sram_act_for_8 - DCNN1_CHIN/2:addr_sram_act_for_8 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_8 <= addr_sram_act_for_8;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_8 <= (cnt_bt_mem[6])? addr_sram_act_for_8 - DCNN1_CHIN/2: addr_sram_act_for_8 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_8 <= (cnt_bt_mem[6])?  addr_sram_act_for_8 - DCNN1_CHIN/2: addr_sram_act_for_8 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_8 <= addr_sram_act_for_8;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_8 <= (cnt_bt_mem[6])?  addr_sram_act_for_8 - DCNN1_CHIN/2: addr_sram_act_for_8 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_8 <= (cnt_bt_mem[6])?  addr_sram_act_for_8 - DCNN1_CHIN/2: addr_sram_act_for_8 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_8 <= addr_sram_act_for_8;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_8 <= (cnt_bt_mem[6])? addr_sram_act_for_8 - DCNN1_CHIN/2: addr_sram_act_for_8 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_8 <= addr_sram_act_for_8;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_8 <= (cnt_bt_mem[6])? addr_sram_act_for_8 - DCNN1_CHIN/2: addr_sram_act_for_8 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_8 <= (cnt_bt_mem[6])? addr_sram_act_for_8 - DCNN1_CHIN/2: addr_sram_act_for_8 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_8 <= addr_sram_act_for_8;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_8 <= (cnt_bt_mem[6])? addr_sram_act_for_8 - DCNN1_CHIN/2:addr_sram_act_for_8 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_8 <= (cnt_bt_mem[6])? addr_sram_act_for_8 - DCNN1_CHIN/2:addr_sram_act_for_8 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_8 <= addr_sram_act_for_8;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_8 <= 1;
                            if (cnt_bt_mem[6] == 0) begin
                                if (padding_crl_mem[6] == 0) addr_sram_act_for_8 <= (is_odd_mem[6])?  addr_sram_act_for_8 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_8 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[6] == 1)  | (padding_crl_mem[6] == 2) | (padding_crl_mem[6] == 3)) addr_sram_act_for_8 <= ADDR_LSTM10_SRAM_ACT_INIT + 7;
                                else if ((padding_crl_mem[6] == 4)  | (padding_crl_mem[6] == 5) ) addr_sram_act_for_8 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 7;
                                else addr_sram_act_for_8 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 7 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[6] == 0) addr_sram_act_for_8 <= (is_odd_mem[6])? addr_sram_act_for_8 + 3*DCNN1_CHIN/2 : addr_sram_act_for_8 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[6] == 1)  | (padding_crl_mem[6] == 2) | (padding_crl_mem[6] == 3)) addr_sram_act_for_8 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 7;
                                else if ((padding_crl_mem[6] == 4)  | (padding_crl_mem[6] == 5)) addr_sram_act_for_8 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 7;
                                else addr_sram_act_for_8 <= ADDR_LSTM10_SRAM_ACT_INIT + 7;
                            end
                        end
                    end
                    else if ((conv_state_mem[7] == load_a) & !addr_sram_act_re_end_9) begin
                        addr_sram_act_re_end_8 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_9 <= 0;
                            case (padding_crl_mem[7])
                                0: begin
                                    if (~is_odd_mem[7]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_9 <= (cnt_bt_mem[7])? addr_sram_act_for_9 - DCNN1_CHIN/2 : addr_sram_act_for_9 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_9 <= addr_sram_act_for_9;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_9 <= (cnt_bt_mem[7])? addr_sram_act_for_9 - DCNN1_CHIN/2 : addr_sram_act_for_9 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_9 <= addr_sram_act_for_9;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_9 <= (cnt_bt_mem[7])? addr_sram_act_for_9 - DCNN1_CHIN/2:addr_sram_act_for_9 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_9 <= addr_sram_act_for_9;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_9 <= (cnt_bt_mem[7])? addr_sram_act_for_9 - DCNN1_CHIN/2: addr_sram_act_for_9 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_9 <= (cnt_bt_mem[7])?  addr_sram_act_for_9 - DCNN1_CHIN/2: addr_sram_act_for_9 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_9 <= addr_sram_act_for_9;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_9 <= (cnt_bt_mem[7])?  addr_sram_act_for_9 - DCNN1_CHIN/2: addr_sram_act_for_9 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_9 <= (cnt_bt_mem[7])?  addr_sram_act_for_9 - DCNN1_CHIN/2: addr_sram_act_for_9 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_9 <= addr_sram_act_for_9;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_9 <= (cnt_bt_mem[7])? addr_sram_act_for_9 - DCNN1_CHIN/2: addr_sram_act_for_9 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_9 <= addr_sram_act_for_9;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_9 <= (cnt_bt_mem[7])? addr_sram_act_for_9 - DCNN1_CHIN/2: addr_sram_act_for_9 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_9 <= (cnt_bt_mem[7])? addr_sram_act_for_9 - DCNN1_CHIN/2: addr_sram_act_for_9 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_9 <= addr_sram_act_for_9;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_9 <= (cnt_bt_mem[7])? addr_sram_act_for_9 - DCNN1_CHIN/2:addr_sram_act_for_9 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_9 <= (cnt_bt_mem[7])? addr_sram_act_for_9 - DCNN1_CHIN/2:addr_sram_act_for_9 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_9 <= addr_sram_act_for_9;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_9 <= 1;
                            if (cnt_bt_mem[7] == 0) begin
                                if (padding_crl_mem[7] == 0) addr_sram_act_for_9 <= (is_odd_mem[7])?  addr_sram_act_for_9 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_9 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[7] == 1)  | (padding_crl_mem[7] == 2) | (padding_crl_mem[7] == 3)) addr_sram_act_for_9 <= ADDR_LSTM10_SRAM_ACT_INIT + 8;
                                else if ((padding_crl_mem[7] == 4)  | (padding_crl_mem[7] == 5) ) addr_sram_act_for_9 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 8;
                                else addr_sram_act_for_9 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 8 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[7] == 0) addr_sram_act_for_9 <= (is_odd_mem[7])? addr_sram_act_for_9 + 3*DCNN1_CHIN/2 : addr_sram_act_for_9 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[7] == 1)  | (padding_crl_mem[7] == 2) | (padding_crl_mem[7] == 3)) addr_sram_act_for_9 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 8;
                                else if ((padding_crl_mem[7] == 4)  | (padding_crl_mem[7] == 5)) addr_sram_act_for_9 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 8;
                                else addr_sram_act_for_9 <= ADDR_LSTM10_SRAM_ACT_INIT + 8;
                            end
                        end
                    end
                    else if ((conv_state_mem[8] == load_a) & !addr_sram_act_re_end_10) begin
                        addr_sram_act_re_end_9 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_10 <= 0;
                            case (padding_crl_mem[8])
                                0: begin
                                    if (~is_odd_mem[8]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_10 <= (cnt_bt_mem[8])? addr_sram_act_for_10 - DCNN1_CHIN/2 : addr_sram_act_for_10 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_10 <= addr_sram_act_for_10;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_10 <= (cnt_bt_mem[8])? addr_sram_act_for_10 - DCNN1_CHIN/2 : addr_sram_act_for_10 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_10 <= addr_sram_act_for_10;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_10 <= (cnt_bt_mem[8])? addr_sram_act_for_10 - DCNN1_CHIN/2:addr_sram_act_for_10 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_10 <= addr_sram_act_for_10;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_10 <= (cnt_bt_mem[8])? addr_sram_act_for_10 - DCNN1_CHIN/2: addr_sram_act_for_10 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_10 <= (cnt_bt_mem[8])?  addr_sram_act_for_10 - DCNN1_CHIN/2: addr_sram_act_for_10 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_10 <= addr_sram_act_for_10;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_10 <= (cnt_bt_mem[8])?  addr_sram_act_for_10 - DCNN1_CHIN/2: addr_sram_act_for_10 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_10 <= (cnt_bt_mem[8])?  addr_sram_act_for_10 - DCNN1_CHIN/2: addr_sram_act_for_10 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_10 <= addr_sram_act_for_10;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_10 <= (cnt_bt_mem[8])? addr_sram_act_for_10 - DCNN1_CHIN/2: addr_sram_act_for_10 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_10 <= addr_sram_act_for_10;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_10 <= (cnt_bt_mem[8])? addr_sram_act_for_10 - DCNN1_CHIN/2: addr_sram_act_for_10 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_10 <= (cnt_bt_mem[8])? addr_sram_act_for_10 - DCNN1_CHIN/2: addr_sram_act_for_10 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_10 <= addr_sram_act_for_10;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_10 <= (cnt_bt_mem[8])? addr_sram_act_for_10 - DCNN1_CHIN/2:addr_sram_act_for_10 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_10 <= (cnt_bt_mem[8])? addr_sram_act_for_10 - DCNN1_CHIN/2:addr_sram_act_for_10 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_10 <= addr_sram_act_for_10;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_10 <= 1;
                            if (cnt_bt_mem[8] == 0) begin
                                if (padding_crl_mem[8] == 0) addr_sram_act_for_10 <= (is_odd_mem[8])?  addr_sram_act_for_10 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_10 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[8] == 1)  | (padding_crl_mem[8] == 2) | (padding_crl_mem[8] == 3)) addr_sram_act_for_10 <= ADDR_LSTM10_SRAM_ACT_INIT + 9;
                                else if ((padding_crl_mem[8] == 4)  | (padding_crl_mem[8] == 5) ) addr_sram_act_for_10 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 9;
                                else addr_sram_act_for_10 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 9 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[8] == 0) addr_sram_act_for_10 <= (is_odd_mem[8])? addr_sram_act_for_10 + 3*DCNN1_CHIN/2 : addr_sram_act_for_10 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[8] == 1)  | (padding_crl_mem[8] == 2) | (padding_crl_mem[8] == 3)) addr_sram_act_for_10 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 9;
                                else if ((padding_crl_mem[8] == 4)  | (padding_crl_mem[8] == 5)) addr_sram_act_for_10 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 9;
                                else addr_sram_act_for_10 <= ADDR_LSTM10_SRAM_ACT_INIT + 9;
                            end
                        end
                    end
                    else if ((conv_state_mem[9] == load_a) & !addr_sram_act_re_end_11) begin
                        addr_sram_act_re_end_10 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_11 <= 0;
                            case (padding_crl_mem[9])
                                0: begin
                                    if (~is_odd_mem[9]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_11 <= (cnt_bt_mem[9])? addr_sram_act_for_11 - DCNN1_CHIN/2 : addr_sram_act_for_11 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_11 <= addr_sram_act_for_11;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_11 <= (cnt_bt_mem[9])? addr_sram_act_for_11 - DCNN1_CHIN/2 : addr_sram_act_for_11 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_11 <= addr_sram_act_for_11;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_11 <= (cnt_bt_mem[9])? addr_sram_act_for_11 - DCNN1_CHIN/2:addr_sram_act_for_11 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_11 <= addr_sram_act_for_11;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_11 <= (cnt_bt_mem[9])? addr_sram_act_for_11 - DCNN1_CHIN/2: addr_sram_act_for_11 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_11 <= (cnt_bt_mem[9])?  addr_sram_act_for_11 - DCNN1_CHIN/2: addr_sram_act_for_11 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_11 <= addr_sram_act_for_11;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_11 <= (cnt_bt_mem[9])?  addr_sram_act_for_11 - DCNN1_CHIN/2: addr_sram_act_for_11 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_11 <= (cnt_bt_mem[9])?  addr_sram_act_for_11 - DCNN1_CHIN/2: addr_sram_act_for_11 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_11 <= addr_sram_act_for_11;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_11 <= (cnt_bt_mem[9])? addr_sram_act_for_11 - DCNN1_CHIN/2: addr_sram_act_for_11 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_11 <= addr_sram_act_for_11;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_11 <= (cnt_bt_mem[9])? addr_sram_act_for_11 - DCNN1_CHIN/2: addr_sram_act_for_11 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_11 <= (cnt_bt_mem[9])? addr_sram_act_for_11 - DCNN1_CHIN/2: addr_sram_act_for_11 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_11 <= addr_sram_act_for_11;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_11 <= (cnt_bt_mem[9])? addr_sram_act_for_11 - DCNN1_CHIN/2:addr_sram_act_for_11 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_11 <= (cnt_bt_mem[9])? addr_sram_act_for_11 - DCNN1_CHIN/2:addr_sram_act_for_11 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_11 <= addr_sram_act_for_11;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_11 <= 1;
                            if (cnt_bt_mem[9] == 0) begin
                                if (padding_crl_mem[9] == 0) addr_sram_act_for_11 <= (is_odd_mem[9])?  addr_sram_act_for_11 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_11 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[9] == 1)  | (padding_crl_mem[9] == 2) | (padding_crl_mem[9] == 3)) addr_sram_act_for_11 <= ADDR_LSTM10_SRAM_ACT_INIT + 10;
                                else if ((padding_crl_mem[9] == 4)  | (padding_crl_mem[9] == 5) ) addr_sram_act_for_11 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 10;
                                else addr_sram_act_for_11 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 10 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[9] == 0) addr_sram_act_for_11 <= (is_odd_mem[9])? addr_sram_act_for_11 + 3*DCNN1_CHIN/2 : addr_sram_act_for_11 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[9] == 1)  | (padding_crl_mem[9] == 2) | (padding_crl_mem[9] == 3)) addr_sram_act_for_11 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 10;
                                else if ((padding_crl_mem[9] == 4)  | (padding_crl_mem[9] == 5)) addr_sram_act_for_11 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 10;
                                else addr_sram_act_for_11 <= ADDR_LSTM10_SRAM_ACT_INIT + 10;
                            end
                        end
                    end
                    else if ((conv_state_mem[10] == load_a) & !addr_sram_act_re_end_12) begin
                        addr_sram_act_re_end_11 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_12 <= 0;
                            case (padding_crl_mem[10])
                                0: begin
                                    if (~is_odd_mem[10]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_12 <= (cnt_bt_mem[10])? addr_sram_act_for_12 - DCNN1_CHIN/2 : addr_sram_act_for_12 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_12 <= addr_sram_act_for_12;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_12 <= (cnt_bt_mem[10])? addr_sram_act_for_12 - DCNN1_CHIN/2 : addr_sram_act_for_12 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_12 <= addr_sram_act_for_12;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_12 <= (cnt_bt_mem[10])? addr_sram_act_for_12 - DCNN1_CHIN/2:addr_sram_act_for_12 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_12 <= addr_sram_act_for_12;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_12 <= (cnt_bt_mem[10])? addr_sram_act_for_12 - DCNN1_CHIN/2: addr_sram_act_for_12 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_12 <= (cnt_bt_mem[10])?  addr_sram_act_for_12 - DCNN1_CHIN/2: addr_sram_act_for_12 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_12 <= addr_sram_act_for_12;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_12 <= (cnt_bt_mem[10])?  addr_sram_act_for_12 - DCNN1_CHIN/2: addr_sram_act_for_12 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_12 <= (cnt_bt_mem[10])?  addr_sram_act_for_12 - DCNN1_CHIN/2: addr_sram_act_for_12 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_12 <= addr_sram_act_for_12;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_12 <= (cnt_bt_mem[10])? addr_sram_act_for_12 - DCNN1_CHIN/2: addr_sram_act_for_12 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_12 <= addr_sram_act_for_12;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_12 <= (cnt_bt_mem[10])? addr_sram_act_for_12 - DCNN1_CHIN/2: addr_sram_act_for_12 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_12 <= (cnt_bt_mem[10])? addr_sram_act_for_12 - DCNN1_CHIN/2: addr_sram_act_for_12 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_12 <= addr_sram_act_for_12;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_12 <= (cnt_bt_mem[10])? addr_sram_act_for_12 - DCNN1_CHIN/2:addr_sram_act_for_12 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_12 <= (cnt_bt_mem[10])? addr_sram_act_for_12 - DCNN1_CHIN/2:addr_sram_act_for_12 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_12 <= addr_sram_act_for_12;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_12 <= 1;
                            if (cnt_bt_mem[10] == 0) begin
                                if (padding_crl_mem[10] == 0) addr_sram_act_for_12 <= (is_odd_mem[10])?  addr_sram_act_for_12 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_12 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[10] == 1)  | (padding_crl_mem[10] == 2) | (padding_crl_mem[10] == 3)) addr_sram_act_for_12 <= ADDR_LSTM10_SRAM_ACT_INIT + 11;
                                else if ((padding_crl_mem[10] == 4)  | (padding_crl_mem[10] == 5) ) addr_sram_act_for_12 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 11;
                                else addr_sram_act_for_12 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 11 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[10] == 0) addr_sram_act_for_12 <= (is_odd_mem[10])? addr_sram_act_for_12 + 3*DCNN1_CHIN/2 : addr_sram_act_for_12 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[10] == 1)  | (padding_crl_mem[10] == 2) | (padding_crl_mem[10] == 3)) addr_sram_act_for_12 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 11;
                                else if ((padding_crl_mem[10] == 4)  | (padding_crl_mem[10] == 5)) addr_sram_act_for_12 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 11;
                                else addr_sram_act_for_12 <= ADDR_LSTM10_SRAM_ACT_INIT + 11;
                            end
                        end
                    end
                    else if ((conv_state_mem[11] == load_a) & !addr_sram_act_re_end_13) begin
                        addr_sram_act_re_end_12 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_13 <= 0;
                            case (padding_crl_mem[11])
                                0: begin
                                    if (~is_odd_mem[11]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_13 <= (cnt_bt_mem[11])? addr_sram_act_for_13 - DCNN1_CHIN/2 : addr_sram_act_for_13 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_13 <= addr_sram_act_for_13;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_13 <= (cnt_bt_mem[11])? addr_sram_act_for_13 - DCNN1_CHIN/2 : addr_sram_act_for_13 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_13 <= addr_sram_act_for_13;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_13 <= (cnt_bt_mem[11])? addr_sram_act_for_13 - DCNN1_CHIN/2:addr_sram_act_for_13 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_13 <= addr_sram_act_for_13;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_13 <= (cnt_bt_mem[11])? addr_sram_act_for_13 - DCNN1_CHIN/2: addr_sram_act_for_13 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_13 <= (cnt_bt_mem[11])?  addr_sram_act_for_13 - DCNN1_CHIN/2: addr_sram_act_for_13 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_13 <= addr_sram_act_for_13;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_13 <= (cnt_bt_mem[11])?  addr_sram_act_for_13 - DCNN1_CHIN/2: addr_sram_act_for_13 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_13 <= (cnt_bt_mem[11])?  addr_sram_act_for_13 - DCNN1_CHIN/2: addr_sram_act_for_13 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_13 <= addr_sram_act_for_13;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_13 <= (cnt_bt_mem[11])? addr_sram_act_for_13 - DCNN1_CHIN/2: addr_sram_act_for_13 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_13 <= addr_sram_act_for_13;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_13 <= (cnt_bt_mem[11])? addr_sram_act_for_13 - DCNN1_CHIN/2: addr_sram_act_for_13 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_13 <= (cnt_bt_mem[11])? addr_sram_act_for_13 - DCNN1_CHIN/2: addr_sram_act_for_13 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_13 <= addr_sram_act_for_13;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_13 <= (cnt_bt_mem[11])? addr_sram_act_for_13 - DCNN1_CHIN/2:addr_sram_act_for_13 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_13 <= (cnt_bt_mem[11])? addr_sram_act_for_13 - DCNN1_CHIN/2:addr_sram_act_for_13 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_13 <= addr_sram_act_for_13;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_13 <= 1;
                            if (cnt_bt_mem[11] == 0) begin
                                if (padding_crl_mem[11] == 0) addr_sram_act_for_13 <= (is_odd_mem[11])?  addr_sram_act_for_13 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_13 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[11] == 1)  | (padding_crl_mem[11] == 2) | (padding_crl_mem[11] == 3)) addr_sram_act_for_13 <= ADDR_LSTM10_SRAM_ACT_INIT + 12;
                                else if ((padding_crl_mem[11] == 4)  | (padding_crl_mem[11] == 5) ) addr_sram_act_for_13 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 12;
                                else addr_sram_act_for_13 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 12 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[11] == 0) addr_sram_act_for_13 <= (is_odd_mem[11])? addr_sram_act_for_13 + 3*DCNN1_CHIN/2 : addr_sram_act_for_13 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[11] == 1)  | (padding_crl_mem[11] == 2) | (padding_crl_mem[11] == 3)) addr_sram_act_for_13 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 12;
                                else if ((padding_crl_mem[11] == 4)  | (padding_crl_mem[11] == 5)) addr_sram_act_for_13 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 12;
                                else addr_sram_act_for_13 <= ADDR_LSTM10_SRAM_ACT_INIT + 12;
                            end
                        end
                    end
                    else if ((conv_state_mem[12] == load_a) & !addr_sram_act_re_end_14) begin
                        addr_sram_act_re_end_13 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_14 <= 0;
                            case (padding_crl_mem[12])
                                0: begin
                                    if (~is_odd_mem[12]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_14 <= (cnt_bt_mem[12])? addr_sram_act_for_14 - DCNN1_CHIN/2 : addr_sram_act_for_14 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_14 <= addr_sram_act_for_14;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_14 <= (cnt_bt_mem[12])? addr_sram_act_for_14 - DCNN1_CHIN/2 : addr_sram_act_for_14 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_14 <= addr_sram_act_for_14;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_14 <= (cnt_bt_mem[12])? addr_sram_act_for_14 - DCNN1_CHIN/2:addr_sram_act_for_14 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_14 <= addr_sram_act_for_14;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_14 <= (cnt_bt_mem[12])? addr_sram_act_for_14 - DCNN1_CHIN/2: addr_sram_act_for_14 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_14 <= (cnt_bt_mem[12])?  addr_sram_act_for_14 - DCNN1_CHIN/2: addr_sram_act_for_14 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_14 <= addr_sram_act_for_14;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_14 <= (cnt_bt_mem[12])?  addr_sram_act_for_14 - DCNN1_CHIN/2: addr_sram_act_for_14 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_14 <= (cnt_bt_mem[12])?  addr_sram_act_for_14 - DCNN1_CHIN/2: addr_sram_act_for_14 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_14 <= addr_sram_act_for_14;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_14 <= (cnt_bt_mem[12])? addr_sram_act_for_14 - DCNN1_CHIN/2: addr_sram_act_for_14 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_14 <= addr_sram_act_for_14;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_14 <= (cnt_bt_mem[12])? addr_sram_act_for_14 - DCNN1_CHIN/2: addr_sram_act_for_14 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_14 <= (cnt_bt_mem[12])? addr_sram_act_for_14 - DCNN1_CHIN/2: addr_sram_act_for_14 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_14 <= addr_sram_act_for_14;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_14 <= (cnt_bt_mem[12])? addr_sram_act_for_14 - DCNN1_CHIN/2:addr_sram_act_for_14 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_14 <= (cnt_bt_mem[12])? addr_sram_act_for_14 - DCNN1_CHIN/2:addr_sram_act_for_14 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_14 <= addr_sram_act_for_14;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_14 <= 1;
                            if (cnt_bt_mem[12] == 0) begin
                                if (padding_crl_mem[12] == 0) addr_sram_act_for_14 <= (is_odd_mem[12])?  addr_sram_act_for_14 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_14 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[12] == 1)  | (padding_crl_mem[12] == 2) | (padding_crl_mem[12] == 3)) addr_sram_act_for_14 <= ADDR_LSTM10_SRAM_ACT_INIT + 13;
                                else if ((padding_crl_mem[12] == 4)  | (padding_crl_mem[12] == 5) ) addr_sram_act_for_14 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 13;
                                else addr_sram_act_for_14 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 13 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[12] == 0) addr_sram_act_for_14 <= (is_odd_mem[12])? addr_sram_act_for_14 + 3*DCNN1_CHIN/2 : addr_sram_act_for_14 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[12] == 1)  | (padding_crl_mem[12] == 2) | (padding_crl_mem[12] == 3)) addr_sram_act_for_14 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 13;
                                else if ((padding_crl_mem[12] == 4)  | (padding_crl_mem[12] == 5)) addr_sram_act_for_14 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 13;
                                else addr_sram_act_for_14 <= ADDR_LSTM10_SRAM_ACT_INIT + 13;
                            end
                        end
                    end
                    else if ((conv_state_mem[13] == load_a) & !addr_sram_act_re_end_15) begin
                        addr_sram_act_re_end_14 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_15 <= 0;
                            case (padding_crl_mem[13])
                                0: begin
                                    if (~is_odd_mem[13]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_15 <= (cnt_bt_mem[13])? addr_sram_act_for_15 - DCNN1_CHIN/2 : addr_sram_act_for_15 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_15 <= addr_sram_act_for_15;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_15 <= (cnt_bt_mem[13])? addr_sram_act_for_15 - DCNN1_CHIN/2 : addr_sram_act_for_15 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_15 <= addr_sram_act_for_15;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_15 <= (cnt_bt_mem[13])? addr_sram_act_for_15 - DCNN1_CHIN/2:addr_sram_act_for_15 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_15 <= addr_sram_act_for_15;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_15 <= (cnt_bt_mem[13])? addr_sram_act_for_15 - DCNN1_CHIN/2: addr_sram_act_for_15 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_15 <= (cnt_bt_mem[13])?  addr_sram_act_for_15 - DCNN1_CHIN/2: addr_sram_act_for_15 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_15 <= addr_sram_act_for_15;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_15 <= (cnt_bt_mem[13])?  addr_sram_act_for_15 - DCNN1_CHIN/2: addr_sram_act_for_15 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_15 <= (cnt_bt_mem[13])?  addr_sram_act_for_15 - DCNN1_CHIN/2: addr_sram_act_for_15 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_15 <= addr_sram_act_for_15;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_15 <= (cnt_bt_mem[13])? addr_sram_act_for_15 - DCNN1_CHIN/2: addr_sram_act_for_15 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_15 <= addr_sram_act_for_15;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_15 <= (cnt_bt_mem[13])? addr_sram_act_for_15 - DCNN1_CHIN/2: addr_sram_act_for_15 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_15 <= (cnt_bt_mem[13])? addr_sram_act_for_15 - DCNN1_CHIN/2: addr_sram_act_for_15 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_15 <= addr_sram_act_for_15;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_15 <= (cnt_bt_mem[13])? addr_sram_act_for_15 - DCNN1_CHIN/2:addr_sram_act_for_15 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_15 <= (cnt_bt_mem[13])? addr_sram_act_for_15 - DCNN1_CHIN/2:addr_sram_act_for_15 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_15 <= addr_sram_act_for_15;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_15 <= 1;
                            if (cnt_bt_mem[13] == 0) begin
                                if (padding_crl_mem[13] == 0) addr_sram_act_for_15 <= (is_odd_mem[13])?  addr_sram_act_for_15 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_15 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[13] == 1)  | (padding_crl_mem[13] == 2) | (padding_crl_mem[13] == 3)) addr_sram_act_for_15 <= ADDR_LSTM10_SRAM_ACT_INIT + 14;
                                else if ((padding_crl_mem[13] == 4)  | (padding_crl_mem[13] == 5) ) addr_sram_act_for_15 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 14;
                                else addr_sram_act_for_15 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 14 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[13] == 0) addr_sram_act_for_15 <= (is_odd_mem[13])? addr_sram_act_for_15 + 3*DCNN1_CHIN/2 : addr_sram_act_for_15 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[13] == 1)  | (padding_crl_mem[13] == 2) | (padding_crl_mem[13] == 3)) addr_sram_act_for_15 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 14;
                                else if ((padding_crl_mem[13] == 4)  | (padding_crl_mem[13] == 5)) addr_sram_act_for_15 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 14;
                                else addr_sram_act_for_15 <= ADDR_LSTM10_SRAM_ACT_INIT + 14;
                            end
                        end
                    end
                    else if ((conv_state_mem[14] == load_a) & !addr_sram_act_re_end_16) begin
                        addr_sram_act_re_end_15 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_16 <= 0;
                            case (padding_crl_mem[14])
                                0: begin
                                    if (~is_odd_mem[14]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_16 <= (cnt_bt_mem[14])? addr_sram_act_for_16 - DCNN1_CHIN/2 : addr_sram_act_for_16 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_16 <= addr_sram_act_for_16;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_16 <= (cnt_bt_mem[14])? addr_sram_act_for_16 - DCNN1_CHIN/2 : addr_sram_act_for_16 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_16 <= addr_sram_act_for_16;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_16 <= (cnt_bt_mem[14])? addr_sram_act_for_16 - DCNN1_CHIN/2:addr_sram_act_for_16 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_16 <= addr_sram_act_for_16;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_16 <= (cnt_bt_mem[14])? addr_sram_act_for_16 - DCNN1_CHIN/2: addr_sram_act_for_16 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_16 <= (cnt_bt_mem[14])?  addr_sram_act_for_16 - DCNN1_CHIN/2: addr_sram_act_for_16 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_16 <= addr_sram_act_for_16;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_16 <= (cnt_bt_mem[14])?  addr_sram_act_for_16 - DCNN1_CHIN/2: addr_sram_act_for_16 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_16 <= (cnt_bt_mem[14])?  addr_sram_act_for_16 - DCNN1_CHIN/2: addr_sram_act_for_16 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_16 <= addr_sram_act_for_16;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_16 <= (cnt_bt_mem[14])? addr_sram_act_for_16 - DCNN1_CHIN/2: addr_sram_act_for_16 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_16 <= addr_sram_act_for_16;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_16 <= (cnt_bt_mem[14])? addr_sram_act_for_16 - DCNN1_CHIN/2: addr_sram_act_for_16 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_16 <= (cnt_bt_mem[14])? addr_sram_act_for_16 - DCNN1_CHIN/2: addr_sram_act_for_16 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_16 <= addr_sram_act_for_16;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_16 <= (cnt_bt_mem[14])? addr_sram_act_for_16 - DCNN1_CHIN/2:addr_sram_act_for_16 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_16 <= (cnt_bt_mem[14])? addr_sram_act_for_16 - DCNN1_CHIN/2:addr_sram_act_for_16 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_16 <= addr_sram_act_for_16;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_16 <= 1;
                            if (cnt_bt_mem[14] == 0) begin
                                if (padding_crl_mem[14] == 0) addr_sram_act_for_16 <= (is_odd_mem[14])?  addr_sram_act_for_16 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_16 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[14] == 1)  | (padding_crl_mem[14] == 2) | (padding_crl_mem[14] == 3)) addr_sram_act_for_16 <= ADDR_LSTM10_SRAM_ACT_INIT + 15;
                                else if ((padding_crl_mem[14] == 4)  | (padding_crl_mem[14] == 5) ) addr_sram_act_for_16 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 15;
                                else addr_sram_act_for_16 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 15 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[14] == 0) addr_sram_act_for_16 <= (is_odd_mem[14])? addr_sram_act_for_16 + 3*DCNN1_CHIN/2 : addr_sram_act_for_16 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[14] == 1)  | (padding_crl_mem[14] == 2) | (padding_crl_mem[14] == 3)) addr_sram_act_for_16 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 15;
                                else if ((padding_crl_mem[14] == 4)  | (padding_crl_mem[14] == 5)) addr_sram_act_for_16 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 15;
                                else addr_sram_act_for_16 <= ADDR_LSTM10_SRAM_ACT_INIT + 15;
                            end
                        end
                    end
                    else if ((conv_state_mem[15] == load_a) & !addr_sram_act_re_end_17) begin
                        addr_sram_act_re_end_16 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_17 <= 0;
                            case (padding_crl_mem[15])
                                0: begin
                                    if (~is_odd_mem[15]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_17 <= (cnt_bt_mem[15])? addr_sram_act_for_17 - DCNN1_CHIN/2 : addr_sram_act_for_17 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_17 <= addr_sram_act_for_17;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_17 <= (cnt_bt_mem[15])? addr_sram_act_for_17 - DCNN1_CHIN/2 : addr_sram_act_for_17 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_17 <= addr_sram_act_for_17;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_17 <= (cnt_bt_mem[15])? addr_sram_act_for_17 - DCNN1_CHIN/2:addr_sram_act_for_17 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_17 <= addr_sram_act_for_17;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_17 <= (cnt_bt_mem[15])? addr_sram_act_for_17 - DCNN1_CHIN/2: addr_sram_act_for_17 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_17 <= (cnt_bt_mem[15])?  addr_sram_act_for_17 - DCNN1_CHIN/2: addr_sram_act_for_17 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_17 <= addr_sram_act_for_17;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_17 <= (cnt_bt_mem[15])?  addr_sram_act_for_17 - DCNN1_CHIN/2: addr_sram_act_for_17 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_17 <= (cnt_bt_mem[15])?  addr_sram_act_for_17 - DCNN1_CHIN/2: addr_sram_act_for_17 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_17 <= addr_sram_act_for_17;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_17 <= (cnt_bt_mem[15])? addr_sram_act_for_17 - DCNN1_CHIN/2: addr_sram_act_for_17 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_17 <= addr_sram_act_for_17;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_17 <= (cnt_bt_mem[15])? addr_sram_act_for_17 - DCNN1_CHIN/2: addr_sram_act_for_17 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_17 <= (cnt_bt_mem[15])? addr_sram_act_for_17 - DCNN1_CHIN/2: addr_sram_act_for_17 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_17 <= addr_sram_act_for_17;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_17 <= (cnt_bt_mem[15])? addr_sram_act_for_17 - DCNN1_CHIN/2:addr_sram_act_for_17 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_17 <= (cnt_bt_mem[15])? addr_sram_act_for_17 - DCNN1_CHIN/2:addr_sram_act_for_17 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_17 <= addr_sram_act_for_17;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_17 <= 1;
                            if (cnt_bt_mem[15] == 0) begin
                                if (padding_crl_mem[15] == 0) addr_sram_act_for_17 <= (is_odd_mem[15])?  addr_sram_act_for_17 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_17 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[15] == 1)  | (padding_crl_mem[15] == 2) | (padding_crl_mem[15] == 3)) addr_sram_act_for_17 <= ADDR_LSTM10_SRAM_ACT_INIT + 16;
                                else if ((padding_crl_mem[15] == 4)  | (padding_crl_mem[15] == 5) ) addr_sram_act_for_17 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 16;
                                else addr_sram_act_for_17 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 16 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[15] == 0) addr_sram_act_for_17 <= (is_odd_mem[15])? addr_sram_act_for_17 + 3*DCNN1_CHIN/2 : addr_sram_act_for_17 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[15] == 1)  | (padding_crl_mem[15] == 2) | (padding_crl_mem[15] == 3)) addr_sram_act_for_17 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 16;
                                else if ((padding_crl_mem[15] == 4)  | (padding_crl_mem[15] == 5)) addr_sram_act_for_17 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 16;
                                else addr_sram_act_for_17 <= ADDR_LSTM10_SRAM_ACT_INIT + 16;
                            end
                        end
                    end
                    else if ((conv_state_mem[16] == load_a) & !addr_sram_act_re_end_18) begin
                        addr_sram_act_re_end_17 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_18 <= 0;
                            case (padding_crl_mem[16])
                                0: begin
                                    if (~is_odd_mem[16]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_18 <= (cnt_bt_mem[16])? addr_sram_act_for_18 - DCNN1_CHIN/2 : addr_sram_act_for_18 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_18 <= addr_sram_act_for_18;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_18 <= (cnt_bt_mem[16])? addr_sram_act_for_18 - DCNN1_CHIN/2 : addr_sram_act_for_18 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_18 <= addr_sram_act_for_18;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_18 <= (cnt_bt_mem[16])? addr_sram_act_for_18 - DCNN1_CHIN/2:addr_sram_act_for_18 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_18 <= addr_sram_act_for_18;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_18 <= (cnt_bt_mem[16])? addr_sram_act_for_18 - DCNN1_CHIN/2: addr_sram_act_for_18 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_18 <= (cnt_bt_mem[16])?  addr_sram_act_for_18 - DCNN1_CHIN/2: addr_sram_act_for_18 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_18 <= addr_sram_act_for_18;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_18 <= (cnt_bt_mem[16])?  addr_sram_act_for_18 - DCNN1_CHIN/2: addr_sram_act_for_18 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_18 <= (cnt_bt_mem[16])?  addr_sram_act_for_18 - DCNN1_CHIN/2: addr_sram_act_for_18 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_18 <= addr_sram_act_for_18;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_18 <= (cnt_bt_mem[16])? addr_sram_act_for_18 - DCNN1_CHIN/2: addr_sram_act_for_18 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_18 <= addr_sram_act_for_18;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_18 <= (cnt_bt_mem[16])? addr_sram_act_for_18 - DCNN1_CHIN/2: addr_sram_act_for_18 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_18 <= (cnt_bt_mem[16])? addr_sram_act_for_18 - DCNN1_CHIN/2: addr_sram_act_for_18 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_18 <= addr_sram_act_for_18;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_18 <= (cnt_bt_mem[16])? addr_sram_act_for_18 - DCNN1_CHIN/2:addr_sram_act_for_18 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_18 <= (cnt_bt_mem[16])? addr_sram_act_for_18 - DCNN1_CHIN/2:addr_sram_act_for_18 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_18 <= addr_sram_act_for_18;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_18 <= 1;
                            if (cnt_bt_mem[16] == 0) begin
                                if (padding_crl_mem[16] == 0) addr_sram_act_for_18 <= (is_odd_mem[16])?  addr_sram_act_for_18 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_18 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[16] == 1)  | (padding_crl_mem[16] == 2) | (padding_crl_mem[16] == 3)) addr_sram_act_for_18 <= ADDR_LSTM10_SRAM_ACT_INIT + 17;
                                else if ((padding_crl_mem[16] == 4)  | (padding_crl_mem[16] == 5) ) addr_sram_act_for_18 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 17;
                                else addr_sram_act_for_18 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 17 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[16] == 0) addr_sram_act_for_18 <= (is_odd_mem[16])? addr_sram_act_for_18 + 3*DCNN1_CHIN/2 : addr_sram_act_for_18 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[16] == 1)  | (padding_crl_mem[16] == 2) | (padding_crl_mem[16] == 3)) addr_sram_act_for_18 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 17;
                                else if ((padding_crl_mem[16] == 4)  | (padding_crl_mem[16] == 5)) addr_sram_act_for_18 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 17;
                                else addr_sram_act_for_18 <= ADDR_LSTM10_SRAM_ACT_INIT + 17;
                            end
                        end
                    end
                    else if ((conv_state_mem[17] == load_a) & !addr_sram_act_re_end_19) begin
                        addr_sram_act_re_end_18 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_19 <= 0;
                            case (padding_crl_mem[17])
                                0: begin
                                    if (~is_odd_mem[17]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_19 <= (cnt_bt_mem[17])? addr_sram_act_for_19 - DCNN1_CHIN/2 : addr_sram_act_for_19 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_19 <= addr_sram_act_for_19;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_19 <= (cnt_bt_mem[17])? addr_sram_act_for_19 - DCNN1_CHIN/2 : addr_sram_act_for_19 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_19 <= addr_sram_act_for_19;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_19 <= (cnt_bt_mem[17])? addr_sram_act_for_19 - DCNN1_CHIN/2:addr_sram_act_for_19 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_19 <= addr_sram_act_for_19;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_19 <= (cnt_bt_mem[17])? addr_sram_act_for_19 - DCNN1_CHIN/2: addr_sram_act_for_19 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_19 <= (cnt_bt_mem[17])?  addr_sram_act_for_19 - DCNN1_CHIN/2: addr_sram_act_for_19 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_19 <= addr_sram_act_for_19;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_19 <= (cnt_bt_mem[17])?  addr_sram_act_for_19 - DCNN1_CHIN/2: addr_sram_act_for_19 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_19 <= (cnt_bt_mem[17])?  addr_sram_act_for_19 - DCNN1_CHIN/2: addr_sram_act_for_19 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_19 <= addr_sram_act_for_19;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_19 <= (cnt_bt_mem[17])? addr_sram_act_for_19 - DCNN1_CHIN/2: addr_sram_act_for_19 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_19 <= addr_sram_act_for_19;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_19 <= (cnt_bt_mem[17])? addr_sram_act_for_19 - DCNN1_CHIN/2: addr_sram_act_for_19 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_19 <= (cnt_bt_mem[17])? addr_sram_act_for_19 - DCNN1_CHIN/2: addr_sram_act_for_19 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_19 <= addr_sram_act_for_19;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_19 <= (cnt_bt_mem[17])? addr_sram_act_for_19 - DCNN1_CHIN/2:addr_sram_act_for_19 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_19 <= (cnt_bt_mem[17])? addr_sram_act_for_19 - DCNN1_CHIN/2:addr_sram_act_for_19 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_19 <= addr_sram_act_for_19;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_19 <= 1;
                            if (cnt_bt_mem[17] == 0) begin
                                if (padding_crl_mem[17] == 0) addr_sram_act_for_19 <= (is_odd_mem[17])?  addr_sram_act_for_19 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_19 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[17] == 1)  | (padding_crl_mem[17] == 2) | (padding_crl_mem[17] == 3)) addr_sram_act_for_19 <= ADDR_LSTM10_SRAM_ACT_INIT + 18;
                                else if ((padding_crl_mem[17] == 4)  | (padding_crl_mem[17] == 5) ) addr_sram_act_for_19 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 18;
                                else addr_sram_act_for_19 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 18 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[17] == 0) addr_sram_act_for_19 <= (is_odd_mem[17])? addr_sram_act_for_19 + 3*DCNN1_CHIN/2 : addr_sram_act_for_19 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[17] == 1)  | (padding_crl_mem[17] == 2) | (padding_crl_mem[17] == 3)) addr_sram_act_for_19 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 18;
                                else if ((padding_crl_mem[17] == 4)  | (padding_crl_mem[17] == 5)) addr_sram_act_for_19 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 18;
                                else addr_sram_act_for_19 <= ADDR_LSTM10_SRAM_ACT_INIT + 18;
                            end
                        end
                    end
                    else if ((conv_state_mem[18] == load_a) & !addr_sram_act_re_end_20) begin
                        addr_sram_act_re_end_19 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_20 <= 0;
                            case (padding_crl_mem[18])
                                0: begin
                                    if (~is_odd_mem[18]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_20 <= (cnt_bt_mem[18])? addr_sram_act_for_20 - DCNN1_CHIN/2 : addr_sram_act_for_20 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_20 <= addr_sram_act_for_20;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_20 <= (cnt_bt_mem[18])? addr_sram_act_for_20 - DCNN1_CHIN/2 : addr_sram_act_for_20 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_20 <= addr_sram_act_for_20;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_20 <= (cnt_bt_mem[18])? addr_sram_act_for_20 - DCNN1_CHIN/2:addr_sram_act_for_20 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_20 <= addr_sram_act_for_20;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_20 <= (cnt_bt_mem[18])? addr_sram_act_for_20 - DCNN1_CHIN/2: addr_sram_act_for_20 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_20 <= (cnt_bt_mem[18])?  addr_sram_act_for_20 - DCNN1_CHIN/2: addr_sram_act_for_20 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_20 <= addr_sram_act_for_20;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_20 <= (cnt_bt_mem[18])?  addr_sram_act_for_20 - DCNN1_CHIN/2: addr_sram_act_for_20 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_20 <= (cnt_bt_mem[18])?  addr_sram_act_for_20 - DCNN1_CHIN/2: addr_sram_act_for_20 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_20 <= addr_sram_act_for_20;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_20 <= (cnt_bt_mem[18])? addr_sram_act_for_20 - DCNN1_CHIN/2: addr_sram_act_for_20 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_20 <= addr_sram_act_for_20;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_20 <= (cnt_bt_mem[18])? addr_sram_act_for_20 - DCNN1_CHIN/2: addr_sram_act_for_20 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_20 <= (cnt_bt_mem[18])? addr_sram_act_for_20 - DCNN1_CHIN/2: addr_sram_act_for_20 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_20 <= addr_sram_act_for_20;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_20 <= (cnt_bt_mem[18])? addr_sram_act_for_20 - DCNN1_CHIN/2:addr_sram_act_for_20 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_20 <= (cnt_bt_mem[18])? addr_sram_act_for_20 - DCNN1_CHIN/2:addr_sram_act_for_20 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_20 <= addr_sram_act_for_20;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_20 <= 1;
                            if (cnt_bt_mem[18] == 0) begin
                                if (padding_crl_mem[18] == 0) addr_sram_act_for_20 <= (is_odd_mem[18])?  addr_sram_act_for_20 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_20 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[18] == 1)  | (padding_crl_mem[18] == 2) | (padding_crl_mem[18] == 3)) addr_sram_act_for_20 <= ADDR_LSTM10_SRAM_ACT_INIT + 19;
                                else if ((padding_crl_mem[18] == 4)  | (padding_crl_mem[18] == 5) ) addr_sram_act_for_20 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 19;
                                else addr_sram_act_for_20 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 19 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[18] == 0) addr_sram_act_for_20 <= (is_odd_mem[18])? addr_sram_act_for_20 + 3*DCNN1_CHIN/2 : addr_sram_act_for_20 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[18] == 1)  | (padding_crl_mem[18] == 2) | (padding_crl_mem[18] == 3)) addr_sram_act_for_20 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 19;
                                else if ((padding_crl_mem[18] == 4)  | (padding_crl_mem[18] == 5)) addr_sram_act_for_20 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 19;
                                else addr_sram_act_for_20 <= ADDR_LSTM10_SRAM_ACT_INIT + 19;
                            end
                        end
                    end
                    else if ((conv_state_mem[19] == load_a) & !addr_sram_act_re_end_21) begin
                        addr_sram_act_re_end_20 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_21 <= 0;
                            case (padding_crl_mem[19])
                                0: begin
                                    if (~is_odd_mem[19]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_21 <= (cnt_bt_mem[19])? addr_sram_act_for_21 - DCNN1_CHIN/2 : addr_sram_act_for_21 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_21 <= addr_sram_act_for_21;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_21 <= (cnt_bt_mem[19])? addr_sram_act_for_21 - DCNN1_CHIN/2 : addr_sram_act_for_21 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_21 <= addr_sram_act_for_21;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_21 <= (cnt_bt_mem[19])? addr_sram_act_for_21 - DCNN1_CHIN/2:addr_sram_act_for_21 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_21 <= addr_sram_act_for_21;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_21 <= (cnt_bt_mem[19])? addr_sram_act_for_21 - DCNN1_CHIN/2: addr_sram_act_for_21 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_21 <= (cnt_bt_mem[19])?  addr_sram_act_for_21 - DCNN1_CHIN/2: addr_sram_act_for_21 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_21 <= addr_sram_act_for_21;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_21 <= (cnt_bt_mem[19])?  addr_sram_act_for_21 - DCNN1_CHIN/2: addr_sram_act_for_21 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_21 <= (cnt_bt_mem[19])?  addr_sram_act_for_21 - DCNN1_CHIN/2: addr_sram_act_for_21 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_21 <= addr_sram_act_for_21;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_21 <= (cnt_bt_mem[19])? addr_sram_act_for_21 - DCNN1_CHIN/2: addr_sram_act_for_21 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_21 <= addr_sram_act_for_21;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_21 <= (cnt_bt_mem[19])? addr_sram_act_for_21 - DCNN1_CHIN/2: addr_sram_act_for_21 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_21 <= (cnt_bt_mem[19])? addr_sram_act_for_21 - DCNN1_CHIN/2: addr_sram_act_for_21 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_21 <= addr_sram_act_for_21;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_21 <= (cnt_bt_mem[19])? addr_sram_act_for_21 - DCNN1_CHIN/2:addr_sram_act_for_21 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_21 <= (cnt_bt_mem[19])? addr_sram_act_for_21 - DCNN1_CHIN/2:addr_sram_act_for_21 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_21 <= addr_sram_act_for_21;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_21 <= 1;
                            if (cnt_bt_mem[19] == 0) begin
                                if (padding_crl_mem[19] == 0) addr_sram_act_for_21 <= (is_odd_mem[19])?  addr_sram_act_for_21 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_21 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[19] == 1)  | (padding_crl_mem[19] == 2) | (padding_crl_mem[19] == 3)) addr_sram_act_for_21 <= ADDR_LSTM10_SRAM_ACT_INIT + 20;
                                else if ((padding_crl_mem[19] == 4)  | (padding_crl_mem[19] == 5) ) addr_sram_act_for_21 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 20;
                                else addr_sram_act_for_21 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 20 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[19] == 0) addr_sram_act_for_21 <= (is_odd_mem[19])? addr_sram_act_for_21 + 3*DCNN1_CHIN/2 : addr_sram_act_for_21 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[19] == 1)  | (padding_crl_mem[19] == 2) | (padding_crl_mem[19] == 3)) addr_sram_act_for_21 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 20;
                                else if ((padding_crl_mem[19] == 4)  | (padding_crl_mem[19] == 5)) addr_sram_act_for_21 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 20;
                                else addr_sram_act_for_21 <= ADDR_LSTM10_SRAM_ACT_INIT + 20;
                            end
                        end
                    end
                    else if ((conv_state_mem[20] == load_a) & !addr_sram_act_re_end_22) begin
                        addr_sram_act_re_end_21 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_22 <= 0;
                            case (padding_crl_mem[20])
                                0: begin
                                    if (~is_odd_mem[20]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_22 <= (cnt_bt_mem[20])? addr_sram_act_for_22 - DCNN1_CHIN/2 : addr_sram_act_for_22 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_22 <= addr_sram_act_for_22;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_22 <= (cnt_bt_mem[20])? addr_sram_act_for_22 - DCNN1_CHIN/2 : addr_sram_act_for_22 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_22 <= addr_sram_act_for_22;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_22 <= (cnt_bt_mem[20])? addr_sram_act_for_22 - DCNN1_CHIN/2:addr_sram_act_for_22 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_22 <= addr_sram_act_for_22;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_22 <= (cnt_bt_mem[20])? addr_sram_act_for_22 - DCNN1_CHIN/2: addr_sram_act_for_22 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_22 <= (cnt_bt_mem[20])?  addr_sram_act_for_22 - DCNN1_CHIN/2: addr_sram_act_for_22 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_22 <= addr_sram_act_for_22;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_22 <= (cnt_bt_mem[20])?  addr_sram_act_for_22 - DCNN1_CHIN/2: addr_sram_act_for_22 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_22 <= (cnt_bt_mem[20])?  addr_sram_act_for_22 - DCNN1_CHIN/2: addr_sram_act_for_22 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_22 <= addr_sram_act_for_22;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_22 <= (cnt_bt_mem[20])? addr_sram_act_for_22 - DCNN1_CHIN/2: addr_sram_act_for_22 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_22 <= addr_sram_act_for_22;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_22 <= (cnt_bt_mem[20])? addr_sram_act_for_22 - DCNN1_CHIN/2: addr_sram_act_for_22 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_22 <= (cnt_bt_mem[20])? addr_sram_act_for_22 - DCNN1_CHIN/2: addr_sram_act_for_22 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_22 <= addr_sram_act_for_22;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_22 <= (cnt_bt_mem[20])? addr_sram_act_for_22 - DCNN1_CHIN/2:addr_sram_act_for_22 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_22 <= (cnt_bt_mem[20])? addr_sram_act_for_22 - DCNN1_CHIN/2:addr_sram_act_for_22 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_22 <= addr_sram_act_for_22;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_22 <= 1;
                            if (cnt_bt_mem[20] == 0) begin
                                if (padding_crl_mem[20] == 0) addr_sram_act_for_22 <= (is_odd_mem[20])?  addr_sram_act_for_22 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_22 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[20] == 1)  | (padding_crl_mem[20] == 2) | (padding_crl_mem[20] == 3)) addr_sram_act_for_22 <= ADDR_LSTM10_SRAM_ACT_INIT + 21;
                                else if ((padding_crl_mem[20] == 4)  | (padding_crl_mem[20] == 5) ) addr_sram_act_for_22 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 21;
                                else addr_sram_act_for_22 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 21 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[20] == 0) addr_sram_act_for_22 <= (is_odd_mem[20])? addr_sram_act_for_22 + 3*DCNN1_CHIN/2 : addr_sram_act_for_22 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[20] == 1)  | (padding_crl_mem[20] == 2) | (padding_crl_mem[20] == 3)) addr_sram_act_for_22 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 21;
                                else if ((padding_crl_mem[20] == 4)  | (padding_crl_mem[20] == 5)) addr_sram_act_for_22 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 21;
                                else addr_sram_act_for_22 <= ADDR_LSTM10_SRAM_ACT_INIT + 21;
                            end
                        end
                    end
                    else if ((conv_state_mem[21] == load_a) & !addr_sram_act_re_end_23) begin
                        addr_sram_act_re_end_22 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_23 <= 0;
                            case (padding_crl_mem[21])
                                0: begin
                                    if (~is_odd_mem[21]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_23 <= (cnt_bt_mem[21])? addr_sram_act_for_23 - DCNN1_CHIN/2 : addr_sram_act_for_23 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_23 <= addr_sram_act_for_23;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_23 <= (cnt_bt_mem[21])? addr_sram_act_for_23 - DCNN1_CHIN/2 : addr_sram_act_for_23 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_23 <= addr_sram_act_for_23;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_23 <= (cnt_bt_mem[21])? addr_sram_act_for_23 - DCNN1_CHIN/2:addr_sram_act_for_23 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_23 <= addr_sram_act_for_23;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_23 <= (cnt_bt_mem[21])? addr_sram_act_for_23 - DCNN1_CHIN/2: addr_sram_act_for_23 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_23 <= (cnt_bt_mem[21])?  addr_sram_act_for_23 - DCNN1_CHIN/2: addr_sram_act_for_23 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_23 <= addr_sram_act_for_23;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_23 <= (cnt_bt_mem[21])?  addr_sram_act_for_23 - DCNN1_CHIN/2: addr_sram_act_for_23 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_23 <= (cnt_bt_mem[21])?  addr_sram_act_for_23 - DCNN1_CHIN/2: addr_sram_act_for_23 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_23 <= addr_sram_act_for_23;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_23 <= (cnt_bt_mem[21])? addr_sram_act_for_23 - DCNN1_CHIN/2: addr_sram_act_for_23 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_23 <= addr_sram_act_for_23;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_23 <= (cnt_bt_mem[21])? addr_sram_act_for_23 - DCNN1_CHIN/2: addr_sram_act_for_23 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_23 <= (cnt_bt_mem[21])? addr_sram_act_for_23 - DCNN1_CHIN/2: addr_sram_act_for_23 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_23 <= addr_sram_act_for_23;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_23 <= (cnt_bt_mem[21])? addr_sram_act_for_23 - DCNN1_CHIN/2:addr_sram_act_for_23 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_23 <= (cnt_bt_mem[21])? addr_sram_act_for_23 - DCNN1_CHIN/2:addr_sram_act_for_23 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_23 <= addr_sram_act_for_23;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_23 <= 1;
                            if (cnt_bt_mem[21] == 0) begin
                                if (padding_crl_mem[21] == 0) addr_sram_act_for_23 <= (is_odd_mem[21])?  addr_sram_act_for_23 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_23 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[21] == 1)  | (padding_crl_mem[21] == 2) | (padding_crl_mem[21] == 3)) addr_sram_act_for_23 <= ADDR_LSTM10_SRAM_ACT_INIT + 22;
                                else if ((padding_crl_mem[21] == 4)  | (padding_crl_mem[21] == 5) ) addr_sram_act_for_23 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 22;
                                else addr_sram_act_for_23 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 22 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[21] == 0) addr_sram_act_for_23 <= (is_odd_mem[21])? addr_sram_act_for_23 + 3*DCNN1_CHIN/2 : addr_sram_act_for_23 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[21] == 1)  | (padding_crl_mem[21] == 2) | (padding_crl_mem[21] == 3)) addr_sram_act_for_23 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 22;
                                else if ((padding_crl_mem[21] == 4)  | (padding_crl_mem[21] == 5)) addr_sram_act_for_23 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 22;
                                else addr_sram_act_for_23 <= ADDR_LSTM10_SRAM_ACT_INIT + 22;
                            end
                        end
                    end
                    else if ((conv_state_mem[22] == load_a) & !addr_sram_act_re_end_24) begin
                        addr_sram_act_re_end_23 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_24 <= 0;
                            case (padding_crl_mem[22])
                                0: begin
                                    if (~is_odd_mem[22]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_24 <= (cnt_bt_mem[22])? addr_sram_act_for_24 - DCNN1_CHIN/2 : addr_sram_act_for_24 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_24 <= addr_sram_act_for_24;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_24 <= (cnt_bt_mem[22])? addr_sram_act_for_24 - DCNN1_CHIN/2 : addr_sram_act_for_24 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_24 <= addr_sram_act_for_24;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_24 <= (cnt_bt_mem[22])? addr_sram_act_for_24 - DCNN1_CHIN/2:addr_sram_act_for_24 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_24 <= addr_sram_act_for_24;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_24 <= (cnt_bt_mem[22])? addr_sram_act_for_24 - DCNN1_CHIN/2: addr_sram_act_for_24 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_24 <= (cnt_bt_mem[22])?  addr_sram_act_for_24 - DCNN1_CHIN/2: addr_sram_act_for_24 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_24 <= addr_sram_act_for_24;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_24 <= (cnt_bt_mem[22])?  addr_sram_act_for_24 - DCNN1_CHIN/2: addr_sram_act_for_24 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_24 <= (cnt_bt_mem[22])?  addr_sram_act_for_24 - DCNN1_CHIN/2: addr_sram_act_for_24 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_24 <= addr_sram_act_for_24;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_24 <= (cnt_bt_mem[22])? addr_sram_act_for_24 - DCNN1_CHIN/2: addr_sram_act_for_24 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_24 <= addr_sram_act_for_24;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_24 <= (cnt_bt_mem[22])? addr_sram_act_for_24 - DCNN1_CHIN/2: addr_sram_act_for_24 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_24 <= (cnt_bt_mem[22])? addr_sram_act_for_24 - DCNN1_CHIN/2: addr_sram_act_for_24 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_24 <= addr_sram_act_for_24;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_24 <= (cnt_bt_mem[22])? addr_sram_act_for_24 - DCNN1_CHIN/2:addr_sram_act_for_24 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_24 <= (cnt_bt_mem[22])? addr_sram_act_for_24 - DCNN1_CHIN/2:addr_sram_act_for_24 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_24 <= addr_sram_act_for_24;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_24 <= 1;
                            if (cnt_bt_mem[22] == 0) begin
                                if (padding_crl_mem[22] == 0) addr_sram_act_for_24 <= (is_odd_mem[22])?  addr_sram_act_for_24 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_24 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[22] == 1)  | (padding_crl_mem[22] == 2) | (padding_crl_mem[22] == 3)) addr_sram_act_for_24 <= ADDR_LSTM10_SRAM_ACT_INIT + 23;
                                else if ((padding_crl_mem[22] == 4)  | (padding_crl_mem[22] == 5) ) addr_sram_act_for_24 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 23;
                                else addr_sram_act_for_24 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 23 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[22] == 0) addr_sram_act_for_24 <= (is_odd_mem[22])? addr_sram_act_for_24 + 3*DCNN1_CHIN/2 : addr_sram_act_for_24 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[22] == 1)  | (padding_crl_mem[22] == 2) | (padding_crl_mem[22] == 3)) addr_sram_act_for_24 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 23;
                                else if ((padding_crl_mem[22] == 4)  | (padding_crl_mem[22] == 5)) addr_sram_act_for_24 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 23;
                                else addr_sram_act_for_24 <= ADDR_LSTM10_SRAM_ACT_INIT + 23;
                            end
                        end
                    end
                    else if ((conv_state_mem[23] == load_a) & !addr_sram_act_re_end_25) begin
                        addr_sram_act_re_end_24 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_25 <= 0;
                            case (padding_crl_mem[23])
                                0: begin
                                    if (~is_odd_mem[23]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_25 <= (cnt_bt_mem[23])? addr_sram_act_for_25 - DCNN1_CHIN/2 : addr_sram_act_for_25 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_25 <= addr_sram_act_for_25;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_25 <= (cnt_bt_mem[23])? addr_sram_act_for_25 - DCNN1_CHIN/2 : addr_sram_act_for_25 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_25 <= addr_sram_act_for_25;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_25 <= (cnt_bt_mem[23])? addr_sram_act_for_25 - DCNN1_CHIN/2:addr_sram_act_for_25 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_25 <= addr_sram_act_for_25;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_25 <= (cnt_bt_mem[23])? addr_sram_act_for_25 - DCNN1_CHIN/2: addr_sram_act_for_25 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_25 <= (cnt_bt_mem[23])?  addr_sram_act_for_25 - DCNN1_CHIN/2: addr_sram_act_for_25 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_25 <= addr_sram_act_for_25;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_25 <= (cnt_bt_mem[23])?  addr_sram_act_for_25 - DCNN1_CHIN/2: addr_sram_act_for_25 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_25 <= (cnt_bt_mem[23])?  addr_sram_act_for_25 - DCNN1_CHIN/2: addr_sram_act_for_25 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_25 <= addr_sram_act_for_25;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_25 <= (cnt_bt_mem[23])? addr_sram_act_for_25 - DCNN1_CHIN/2: addr_sram_act_for_25 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_25 <= addr_sram_act_for_25;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_25 <= (cnt_bt_mem[23])? addr_sram_act_for_25 - DCNN1_CHIN/2: addr_sram_act_for_25 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_25 <= (cnt_bt_mem[23])? addr_sram_act_for_25 - DCNN1_CHIN/2: addr_sram_act_for_25 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_25 <= addr_sram_act_for_25;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_25 <= (cnt_bt_mem[23])? addr_sram_act_for_25 - DCNN1_CHIN/2:addr_sram_act_for_25 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_25 <= (cnt_bt_mem[23])? addr_sram_act_for_25 - DCNN1_CHIN/2:addr_sram_act_for_25 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_25 <= addr_sram_act_for_25;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_25 <= 1;
                            if (cnt_bt_mem[23] == 0) begin
                                if (padding_crl_mem[23] == 0) addr_sram_act_for_25 <= (is_odd_mem[23])?  addr_sram_act_for_25 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_25 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[23] == 1)  | (padding_crl_mem[23] == 2) | (padding_crl_mem[23] == 3)) addr_sram_act_for_25 <= ADDR_LSTM10_SRAM_ACT_INIT + 24;
                                else if ((padding_crl_mem[23] == 4)  | (padding_crl_mem[23] == 5) ) addr_sram_act_for_25 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 24;
                                else addr_sram_act_for_25 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 24 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[23] == 0) addr_sram_act_for_25 <= (is_odd_mem[23])? addr_sram_act_for_25 + 3*DCNN1_CHIN/2 : addr_sram_act_for_25 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[23] == 1)  | (padding_crl_mem[23] == 2) | (padding_crl_mem[23] == 3)) addr_sram_act_for_25 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 24;
                                else if ((padding_crl_mem[23] == 4)  | (padding_crl_mem[23] == 5)) addr_sram_act_for_25 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 24;
                                else addr_sram_act_for_25 <= ADDR_LSTM10_SRAM_ACT_INIT + 24;
                            end
                        end
                    end
                    else if ((conv_state_mem[24] == load_a) & !addr_sram_act_re_end_26) begin
                        addr_sram_act_re_end_25 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_26 <= 0;
                            case (padding_crl_mem[24])
                                0: begin
                                    if (~is_odd_mem[24]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_26 <= (cnt_bt_mem[24])? addr_sram_act_for_26 - DCNN1_CHIN/2 : addr_sram_act_for_26 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_26 <= addr_sram_act_for_26;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_26 <= (cnt_bt_mem[24])? addr_sram_act_for_26 - DCNN1_CHIN/2 : addr_sram_act_for_26 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_26 <= addr_sram_act_for_26;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_26 <= (cnt_bt_mem[24])? addr_sram_act_for_26 - DCNN1_CHIN/2:addr_sram_act_for_26 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_26 <= addr_sram_act_for_26;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_26 <= (cnt_bt_mem[24])? addr_sram_act_for_26 - DCNN1_CHIN/2: addr_sram_act_for_26 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_26 <= (cnt_bt_mem[24])?  addr_sram_act_for_26 - DCNN1_CHIN/2: addr_sram_act_for_26 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_26 <= addr_sram_act_for_26;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_26 <= (cnt_bt_mem[24])?  addr_sram_act_for_26 - DCNN1_CHIN/2: addr_sram_act_for_26 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_26 <= (cnt_bt_mem[24])?  addr_sram_act_for_26 - DCNN1_CHIN/2: addr_sram_act_for_26 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_26 <= addr_sram_act_for_26;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_26 <= (cnt_bt_mem[24])? addr_sram_act_for_26 - DCNN1_CHIN/2: addr_sram_act_for_26 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_26 <= addr_sram_act_for_26;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_26 <= (cnt_bt_mem[24])? addr_sram_act_for_26 - DCNN1_CHIN/2: addr_sram_act_for_26 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_26 <= (cnt_bt_mem[24])? addr_sram_act_for_26 - DCNN1_CHIN/2: addr_sram_act_for_26 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_26 <= addr_sram_act_for_26;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_26 <= (cnt_bt_mem[24])? addr_sram_act_for_26 - DCNN1_CHIN/2:addr_sram_act_for_26 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_26 <= (cnt_bt_mem[24])? addr_sram_act_for_26 - DCNN1_CHIN/2:addr_sram_act_for_26 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_26 <= addr_sram_act_for_26;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_26 <= 1;
                            if (cnt_bt_mem[24] == 0) begin
                                if (padding_crl_mem[24] == 0) addr_sram_act_for_26 <= (is_odd_mem[24])?  addr_sram_act_for_26 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_26 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[24] == 1)  | (padding_crl_mem[24] == 2) | (padding_crl_mem[24] == 3)) addr_sram_act_for_26 <= ADDR_LSTM10_SRAM_ACT_INIT + 25;
                                else if ((padding_crl_mem[24] == 4)  | (padding_crl_mem[24] == 5) ) addr_sram_act_for_26 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 25;
                                else addr_sram_act_for_26 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 25 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[24] == 0) addr_sram_act_for_26 <= (is_odd_mem[24])? addr_sram_act_for_26 + 3*DCNN1_CHIN/2 : addr_sram_act_for_26 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[24] == 1)  | (padding_crl_mem[24] == 2) | (padding_crl_mem[24] == 3)) addr_sram_act_for_26 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 25;
                                else if ((padding_crl_mem[24] == 4)  | (padding_crl_mem[24] == 5)) addr_sram_act_for_26 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 25;
                                else addr_sram_act_for_26 <= ADDR_LSTM10_SRAM_ACT_INIT + 25;
                            end
                        end
                    end
                    else if ((conv_state_mem[25] == load_a) & !addr_sram_act_re_end_27) begin
                        addr_sram_act_re_end_26 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_27 <= 0;
                            case (padding_crl_mem[25])
                                0: begin
                                    if (~is_odd_mem[25]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_27 <= (cnt_bt_mem[25])? addr_sram_act_for_27 - DCNN1_CHIN/2 : addr_sram_act_for_27 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_27 <= addr_sram_act_for_27;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_27 <= (cnt_bt_mem[25])? addr_sram_act_for_27 - DCNN1_CHIN/2 : addr_sram_act_for_27 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_27 <= addr_sram_act_for_27;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_27 <= (cnt_bt_mem[25])? addr_sram_act_for_27 - DCNN1_CHIN/2:addr_sram_act_for_27 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_27 <= addr_sram_act_for_27;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_27 <= (cnt_bt_mem[25])? addr_sram_act_for_27 - DCNN1_CHIN/2: addr_sram_act_for_27 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_27 <= (cnt_bt_mem[25])?  addr_sram_act_for_27 - DCNN1_CHIN/2: addr_sram_act_for_27 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_27 <= addr_sram_act_for_27;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_27 <= (cnt_bt_mem[25])?  addr_sram_act_for_27 - DCNN1_CHIN/2: addr_sram_act_for_27 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_27 <= (cnt_bt_mem[25])?  addr_sram_act_for_27 - DCNN1_CHIN/2: addr_sram_act_for_27 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_27 <= addr_sram_act_for_27;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_27 <= (cnt_bt_mem[25])? addr_sram_act_for_27 - DCNN1_CHIN/2: addr_sram_act_for_27 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_27 <= addr_sram_act_for_27;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_27 <= (cnt_bt_mem[25])? addr_sram_act_for_27 - DCNN1_CHIN/2: addr_sram_act_for_27 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_27 <= (cnt_bt_mem[25])? addr_sram_act_for_27 - DCNN1_CHIN/2: addr_sram_act_for_27 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_27 <= addr_sram_act_for_27;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_27 <= (cnt_bt_mem[25])? addr_sram_act_for_27 - DCNN1_CHIN/2:addr_sram_act_for_27 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_27 <= (cnt_bt_mem[25])? addr_sram_act_for_27 - DCNN1_CHIN/2:addr_sram_act_for_27 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_27 <= addr_sram_act_for_27;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_27 <= 1;
                            if (cnt_bt_mem[25] == 0) begin
                                if (padding_crl_mem[25] == 0) addr_sram_act_for_27 <= (is_odd_mem[25])?  addr_sram_act_for_27 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_27 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[25] == 1)  | (padding_crl_mem[25] == 2) | (padding_crl_mem[25] == 3)) addr_sram_act_for_27 <= ADDR_LSTM10_SRAM_ACT_INIT + 26;
                                else if ((padding_crl_mem[25] == 4)  | (padding_crl_mem[25] == 5) ) addr_sram_act_for_27 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 26;
                                else addr_sram_act_for_27 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 26 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[25] == 0) addr_sram_act_for_27 <= (is_odd_mem[25])? addr_sram_act_for_27 + 3*DCNN1_CHIN/2 : addr_sram_act_for_27 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[25] == 1)  | (padding_crl_mem[25] == 2) | (padding_crl_mem[25] == 3)) addr_sram_act_for_27 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 26;
                                else if ((padding_crl_mem[25] == 4)  | (padding_crl_mem[25] == 5)) addr_sram_act_for_27 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 26;
                                else addr_sram_act_for_27 <= ADDR_LSTM10_SRAM_ACT_INIT + 26;
                            end
                        end
                    end
                    else if ((conv_state_mem[26] == load_a) & !addr_sram_act_re_end_28) begin
                        addr_sram_act_re_end_27 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_28 <= 0;
                            case (padding_crl_mem[26])
                                0: begin
                                    if (~is_odd_mem[26]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_28 <= (cnt_bt_mem[26])? addr_sram_act_for_28 - DCNN1_CHIN/2 : addr_sram_act_for_28 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_28 <= addr_sram_act_for_28;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_28 <= (cnt_bt_mem[26])? addr_sram_act_for_28 - DCNN1_CHIN/2 : addr_sram_act_for_28 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_28 <= addr_sram_act_for_28;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_28 <= (cnt_bt_mem[26])? addr_sram_act_for_28 - DCNN1_CHIN/2:addr_sram_act_for_28 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_28 <= addr_sram_act_for_28;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_28 <= (cnt_bt_mem[26])? addr_sram_act_for_28 - DCNN1_CHIN/2: addr_sram_act_for_28 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_28 <= (cnt_bt_mem[26])?  addr_sram_act_for_28 - DCNN1_CHIN/2: addr_sram_act_for_28 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_28 <= addr_sram_act_for_28;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_28 <= (cnt_bt_mem[26])?  addr_sram_act_for_28 - DCNN1_CHIN/2: addr_sram_act_for_28 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_28 <= (cnt_bt_mem[26])?  addr_sram_act_for_28 - DCNN1_CHIN/2: addr_sram_act_for_28 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_28 <= addr_sram_act_for_28;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_28 <= (cnt_bt_mem[26])? addr_sram_act_for_28 - DCNN1_CHIN/2: addr_sram_act_for_28 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_28 <= addr_sram_act_for_28;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_28 <= (cnt_bt_mem[26])? addr_sram_act_for_28 - DCNN1_CHIN/2: addr_sram_act_for_28 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_28 <= (cnt_bt_mem[26])? addr_sram_act_for_28 - DCNN1_CHIN/2: addr_sram_act_for_28 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_28 <= addr_sram_act_for_28;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_28 <= (cnt_bt_mem[26])? addr_sram_act_for_28 - DCNN1_CHIN/2:addr_sram_act_for_28 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_28 <= (cnt_bt_mem[26])? addr_sram_act_for_28 - DCNN1_CHIN/2:addr_sram_act_for_28 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_28 <= addr_sram_act_for_28;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_28 <= 1;
                            if (cnt_bt_mem[26] == 0) begin
                                if (padding_crl_mem[26] == 0) addr_sram_act_for_28 <= (is_odd_mem[26])?  addr_sram_act_for_28 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_28 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[26] == 1)  | (padding_crl_mem[26] == 2) | (padding_crl_mem[26] == 3)) addr_sram_act_for_28 <= ADDR_LSTM10_SRAM_ACT_INIT + 27;
                                else if ((padding_crl_mem[26] == 4)  | (padding_crl_mem[26] == 5) ) addr_sram_act_for_28 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 27;
                                else addr_sram_act_for_28 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 27 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[26] == 0) addr_sram_act_for_28 <= (is_odd_mem[26])? addr_sram_act_for_28 + 3*DCNN1_CHIN/2 : addr_sram_act_for_28 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[26] == 1)  | (padding_crl_mem[26] == 2) | (padding_crl_mem[26] == 3)) addr_sram_act_for_28 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 27;
                                else if ((padding_crl_mem[26] == 4)  | (padding_crl_mem[26] == 5)) addr_sram_act_for_28 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 27;
                                else addr_sram_act_for_28 <= ADDR_LSTM10_SRAM_ACT_INIT + 27;
                            end
                        end
                    end
                    else if ((conv_state_mem[27] == load_a) & !addr_sram_act_re_end_29) begin
                        addr_sram_act_re_end_28 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_29 <= 0;
                            case (padding_crl_mem[27])
                                0: begin
                                    if (~is_odd_mem[27]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_29 <= (cnt_bt_mem[27])? addr_sram_act_for_29 - DCNN1_CHIN/2 : addr_sram_act_for_29 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_29 <= addr_sram_act_for_29;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_29 <= (cnt_bt_mem[27])? addr_sram_act_for_29 - DCNN1_CHIN/2 : addr_sram_act_for_29 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_29 <= addr_sram_act_for_29;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_29 <= (cnt_bt_mem[27])? addr_sram_act_for_29 - DCNN1_CHIN/2:addr_sram_act_for_29 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_29 <= addr_sram_act_for_29;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_29 <= (cnt_bt_mem[27])? addr_sram_act_for_29 - DCNN1_CHIN/2: addr_sram_act_for_29 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_29 <= (cnt_bt_mem[27])?  addr_sram_act_for_29 - DCNN1_CHIN/2: addr_sram_act_for_29 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_29 <= addr_sram_act_for_29;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_29 <= (cnt_bt_mem[27])?  addr_sram_act_for_29 - DCNN1_CHIN/2: addr_sram_act_for_29 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_29 <= (cnt_bt_mem[27])?  addr_sram_act_for_29 - DCNN1_CHIN/2: addr_sram_act_for_29 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_29 <= addr_sram_act_for_29;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_29 <= (cnt_bt_mem[27])? addr_sram_act_for_29 - DCNN1_CHIN/2: addr_sram_act_for_29 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_29 <= addr_sram_act_for_29;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_29 <= (cnt_bt_mem[27])? addr_sram_act_for_29 - DCNN1_CHIN/2: addr_sram_act_for_29 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_29 <= (cnt_bt_mem[27])? addr_sram_act_for_29 - DCNN1_CHIN/2: addr_sram_act_for_29 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_29 <= addr_sram_act_for_29;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_29 <= (cnt_bt_mem[27])? addr_sram_act_for_29 - DCNN1_CHIN/2:addr_sram_act_for_29 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_29 <= (cnt_bt_mem[27])? addr_sram_act_for_29 - DCNN1_CHIN/2:addr_sram_act_for_29 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_29 <= addr_sram_act_for_29;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_29 <= 1;
                            if (cnt_bt_mem[27] == 0) begin
                                if (padding_crl_mem[27] == 0) addr_sram_act_for_29 <= (is_odd_mem[27])?  addr_sram_act_for_29 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_29 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[27] == 1)  | (padding_crl_mem[27] == 2) | (padding_crl_mem[27] == 3)) addr_sram_act_for_29 <= ADDR_LSTM10_SRAM_ACT_INIT + 28;
                                else if ((padding_crl_mem[27] == 4)  | (padding_crl_mem[27] == 5) ) addr_sram_act_for_29 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 28;
                                else addr_sram_act_for_29 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 28 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[27] == 0) addr_sram_act_for_29 <= (is_odd_mem[27])? addr_sram_act_for_29 + 3*DCNN1_CHIN/2 : addr_sram_act_for_29 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[27] == 1)  | (padding_crl_mem[27] == 2) | (padding_crl_mem[27] == 3)) addr_sram_act_for_29 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 28;
                                else if ((padding_crl_mem[27] == 4)  | (padding_crl_mem[27] == 5)) addr_sram_act_for_29 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 28;
                                else addr_sram_act_for_29 <= ADDR_LSTM10_SRAM_ACT_INIT + 28;
                            end
                        end
                    end
                    else if ((conv_state_mem[28] == load_a) & !addr_sram_act_re_end_30) begin
                        addr_sram_act_re_end_29 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_30 <= 0;
                            case (padding_crl_mem[28])
                                0: begin
                                    if (~is_odd_mem[28]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_30 <= (cnt_bt_mem[28])? addr_sram_act_for_30 - DCNN1_CHIN/2 : addr_sram_act_for_30 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_30 <= addr_sram_act_for_30;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_30 <= (cnt_bt_mem[28])? addr_sram_act_for_30 - DCNN1_CHIN/2 : addr_sram_act_for_30 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_30 <= addr_sram_act_for_30;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_30 <= (cnt_bt_mem[28])? addr_sram_act_for_30 - DCNN1_CHIN/2:addr_sram_act_for_30 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_30 <= addr_sram_act_for_30;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_30 <= (cnt_bt_mem[28])? addr_sram_act_for_30 - DCNN1_CHIN/2: addr_sram_act_for_30 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_30 <= (cnt_bt_mem[28])?  addr_sram_act_for_30 - DCNN1_CHIN/2: addr_sram_act_for_30 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_30 <= addr_sram_act_for_30;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_30 <= (cnt_bt_mem[28])?  addr_sram_act_for_30 - DCNN1_CHIN/2: addr_sram_act_for_30 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_30 <= (cnt_bt_mem[28])?  addr_sram_act_for_30 - DCNN1_CHIN/2: addr_sram_act_for_30 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_30 <= addr_sram_act_for_30;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_30 <= (cnt_bt_mem[28])? addr_sram_act_for_30 - DCNN1_CHIN/2: addr_sram_act_for_30 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_30 <= addr_sram_act_for_30;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_30 <= (cnt_bt_mem[28])? addr_sram_act_for_30 - DCNN1_CHIN/2: addr_sram_act_for_30 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_30 <= (cnt_bt_mem[28])? addr_sram_act_for_30 - DCNN1_CHIN/2: addr_sram_act_for_30 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_30 <= addr_sram_act_for_30;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_30 <= (cnt_bt_mem[28])? addr_sram_act_for_30 - DCNN1_CHIN/2:addr_sram_act_for_30 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_30 <= (cnt_bt_mem[28])? addr_sram_act_for_30 - DCNN1_CHIN/2:addr_sram_act_for_30 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_30 <= addr_sram_act_for_30;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_30 <= 1;
                            if (cnt_bt_mem[28] == 0) begin
                                if (padding_crl_mem[28] == 0) addr_sram_act_for_30 <= (is_odd_mem[28])?  addr_sram_act_for_30 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_30 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[28] == 1)  | (padding_crl_mem[28] == 2) | (padding_crl_mem[28] == 3)) addr_sram_act_for_30 <= ADDR_LSTM10_SRAM_ACT_INIT + 29;
                                else if ((padding_crl_mem[28] == 4)  | (padding_crl_mem[28] == 5) ) addr_sram_act_for_30 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 29;
                                else addr_sram_act_for_30 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 29 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[28] == 0) addr_sram_act_for_30 <= (is_odd_mem[28])? addr_sram_act_for_30 + 3*DCNN1_CHIN/2 : addr_sram_act_for_30 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[28] == 1)  | (padding_crl_mem[28] == 2) | (padding_crl_mem[28] == 3)) addr_sram_act_for_30 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 29;
                                else if ((padding_crl_mem[28] == 4)  | (padding_crl_mem[28] == 5)) addr_sram_act_for_30 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 29;
                                else addr_sram_act_for_30 <= ADDR_LSTM10_SRAM_ACT_INIT + 29;
                            end
                        end
                    end
                    else if ((conv_state_mem[29] == load_a) & !addr_sram_act_re_end_31) begin
                        addr_sram_act_re_end_32 <= 0;
                        addr_sram_act_re_end_30 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_31 <= 0;
                            case (padding_crl_mem[29])
                                0: begin
                                    if (~is_odd_mem[29]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_31 <= (cnt_bt_mem[29])? addr_sram_act_for_31 - DCNN1_CHIN/2 : addr_sram_act_for_31 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_31 <= addr_sram_act_for_31;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_31 <= (cnt_bt_mem[29])? addr_sram_act_for_31 - DCNN1_CHIN/2 : addr_sram_act_for_31 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_31 <= addr_sram_act_for_31;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_31 <= (cnt_bt_mem[29])? addr_sram_act_for_31 - DCNN1_CHIN/2:addr_sram_act_for_31 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_31 <= addr_sram_act_for_31;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_31 <= (cnt_bt_mem[29])? addr_sram_act_for_31 - DCNN1_CHIN/2: addr_sram_act_for_31 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_31 <= (cnt_bt_mem[29])?  addr_sram_act_for_31 - DCNN1_CHIN/2: addr_sram_act_for_31 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_31 <= addr_sram_act_for_31;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_31 <= (cnt_bt_mem[29])?  addr_sram_act_for_31 - DCNN1_CHIN/2: addr_sram_act_for_31 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_31 <= (cnt_bt_mem[29])?  addr_sram_act_for_31 - DCNN1_CHIN/2: addr_sram_act_for_31 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_31 <= addr_sram_act_for_31;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_31 <= (cnt_bt_mem[29])? addr_sram_act_for_31 - DCNN1_CHIN/2: addr_sram_act_for_31 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_31 <= addr_sram_act_for_31;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_31 <= (cnt_bt_mem[29])? addr_sram_act_for_31 - DCNN1_CHIN/2: addr_sram_act_for_31 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_31 <= (cnt_bt_mem[29])? addr_sram_act_for_31 - DCNN1_CHIN/2: addr_sram_act_for_31 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_31 <= addr_sram_act_for_31;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_31 <= (cnt_bt_mem[29])? addr_sram_act_for_31 - DCNN1_CHIN/2:addr_sram_act_for_31 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_31 <= (cnt_bt_mem[29])? addr_sram_act_for_31 - DCNN1_CHIN/2:addr_sram_act_for_31 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_31 <= addr_sram_act_for_31;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_31 <= 1;
                            if (cnt_bt_mem[29] == 0) begin
                                if (padding_crl_mem[29] == 0) addr_sram_act_for_31 <= (is_odd_mem[29])?  addr_sram_act_for_31 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_31 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[29] == 1)  | (padding_crl_mem[29] == 2) | (padding_crl_mem[29] == 3)) addr_sram_act_for_31 <= ADDR_LSTM10_SRAM_ACT_INIT + 30;
                                else if ((padding_crl_mem[29] == 4)  | (padding_crl_mem[29] == 5) ) addr_sram_act_for_31 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 30;
                                else addr_sram_act_for_31 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 30 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[29] == 0) addr_sram_act_for_31 <= (is_odd_mem[29])? addr_sram_act_for_31 + 3*DCNN1_CHIN/2 : addr_sram_act_for_31 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[29] == 1)  | (padding_crl_mem[29] == 2) | (padding_crl_mem[29] == 3)) addr_sram_act_for_31 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 30;
                                else if ((padding_crl_mem[29] == 4)  | (padding_crl_mem[29] == 5)) addr_sram_act_for_31 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 30;
                                else addr_sram_act_for_31 <= ADDR_LSTM10_SRAM_ACT_INIT + 30;
                            end
                        end
                    end
                    else if ((conv_state_mem[30] == load_a) & !addr_sram_act_re_end_32) begin
                        addr_sram_act_re_end_31 <= 0;
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_32 <= 0;
                            case (padding_crl_mem[30])
                                0: begin
                                    if (~is_odd_mem[30]) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))
                                            addr_sram_act_for_32 <= (cnt_bt_mem[30])? addr_sram_act_for_32 - DCNN1_CHIN/2 : addr_sram_act_for_32 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_32 <= addr_sram_act_for_32;
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6))
                                            addr_sram_act_for_32 <= (cnt_bt_mem[30])? addr_sram_act_for_32 - DCNN1_CHIN/2 : addr_sram_act_for_32 + DCNN1_CHIN/2;
                                        else addr_sram_act_for_32 <= addr_sram_act_for_32;
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_32 <= (cnt_bt_mem[30])? addr_sram_act_for_32 - DCNN1_CHIN/2:addr_sram_act_for_32 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_32 <= addr_sram_act_for_32;
                                    end
                                end
                                2:begin
                                    if (cnt_re_sram == 4) begin
                                        addr_sram_act_for_32 <= (cnt_bt_mem[30])? addr_sram_act_for_32 - DCNN1_CHIN/2: addr_sram_act_for_32 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 6) begin
                                        addr_sram_act_for_32 <= (cnt_bt_mem[30])?  addr_sram_act_for_32 - DCNN1_CHIN/2: addr_sram_act_for_32 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_32 <= addr_sram_act_for_32;
                                    end
                                end
                                3:begin
                                    if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_32 <= (cnt_bt_mem[30])?  addr_sram_act_for_32 - DCNN1_CHIN/2: addr_sram_act_for_32 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_32 <= (cnt_bt_mem[30])?  addr_sram_act_for_32 - DCNN1_CHIN/2: addr_sram_act_for_32 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_32 <= addr_sram_act_for_32;
                                    end
                                end
                                4:begin
                                    if (cnt_re_sram == 2|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_32 <= (cnt_bt_mem[30])? addr_sram_act_for_32 - DCNN1_CHIN/2: addr_sram_act_for_32 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_32 <= addr_sram_act_for_32;
                                    end
                                end
                                5:begin
                                    if (cnt_re_sram == 1) begin
                                        addr_sram_act_for_32 <= (cnt_bt_mem[30])? addr_sram_act_for_32 - DCNN1_CHIN/2: addr_sram_act_for_32 + DCNN1_CHIN/2;
                                    end 
                                    else if (cnt_re_sram == 3) begin
                                        addr_sram_act_for_32 <= (cnt_bt_mem[30])? addr_sram_act_for_32 - DCNN1_CHIN/2: addr_sram_act_for_32 + DCNN1_CHIN/2;
                                    end 
                                    else begin
                                        addr_sram_act_for_32 <= addr_sram_act_for_32;
                                    end
                                end
                                6: begin
                                    if (cnt_re_sram == 0) begin
                                        addr_sram_act_for_32 <= (cnt_bt_mem[30])? addr_sram_act_for_32 - DCNN1_CHIN/2:addr_sram_act_for_32 + DCNN1_CHIN/2;
                                    end
                                    else if (cnt_re_sram == 2) begin
                                        addr_sram_act_for_32 <= (cnt_bt_mem[30])? addr_sram_act_for_32 - DCNN1_CHIN/2:addr_sram_act_for_32 + DCNN1_CHIN/2;
                                    end
                                    else begin
                                        addr_sram_act_for_32 <= addr_sram_act_for_32;
                                    end
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_32 <= 1;
                            if (cnt_bt_mem[30] == 0) begin
                                if (padding_crl_mem[30] == 0) addr_sram_act_for_32 <= (is_odd_mem[30])?  addr_sram_act_for_32 - 3*DCNN1_CHIN/2 :  addr_sram_act_for_32 - 3*DCNN1_CHIN/2 + DCNN1_CHIN/2;
                                else if ((padding_crl_mem[30] == 1)  | (padding_crl_mem[30] == 2) | (padding_crl_mem[30] == 3)) addr_sram_act_for_32 <= ADDR_LSTM10_SRAM_ACT_INIT + 31;
                                else if ((padding_crl_mem[30] == 4)  | (padding_crl_mem[30] == 5) ) addr_sram_act_for_32 <= ADDR_LSTM10_SRAM_ACT_INIT + 61*DCNN1_CHIN/2 + 31;
                                else addr_sram_act_for_32 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 31 ; //for bt = 1
                            end
                            else begin
                                if (padding_crl_mem[30] == 0) addr_sram_act_for_32 <= (is_odd_mem[30])? addr_sram_act_for_32 + 3*DCNN1_CHIN/2 : addr_sram_act_for_32 + 3*DCNN1_CHIN/2 - DCNN1_CHIN/2 ;
                                else if ((padding_crl_mem[30] == 1)  | (padding_crl_mem[30] == 2) | (padding_crl_mem[30] == 3)) addr_sram_act_for_32 <= ADDR_LSTM11_SRAM_ACT_INIT +  (DCNN1_LENGTH_IN-1)*DCNN1_CHIN/2 + 31;
                                else if ((padding_crl_mem[30] == 4)  | (padding_crl_mem[30] == 5)) addr_sram_act_for_32 <= ADDR_LSTM11_SRAM_ACT_INIT +  2*DCNN1_CHIN/2 + 31;
                                else addr_sram_act_for_32 <= ADDR_LSTM10_SRAM_ACT_INIT + 31;
                            end
                        end
                    end
                    else if (layer_done) begin
                        addr_sram_act_for_1 <= ADDR_DCNN1_SRAM_ACT_INIT;
                        addr_sram_act_for_2 <= ADDR_DCNN1_SRAM_ACT_INIT + CNN11_LENGTH_IN;
                        addr_sram_act_for_3 <= ADDR_DCNN1_SRAM_ACT_INIT + 2 * CNN11_LENGTH_IN;
                        addr_sram_act_for_4 <= ADDR_DCNN1_SRAM_ACT_INIT + 3 * CNN11_LENGTH_IN;
                        addr_sram_act_for_5 <= ADDR_DCNN1_SRAM_ACT_INIT + 4 * CNN11_LENGTH_IN;
                        addr_sram_act_for_6 <= ADDR_DCNN1_SRAM_ACT_INIT + 5 * CNN11_LENGTH_IN;
                        addr_sram_act_for_7 <= ADDR_DCNN1_SRAM_ACT_INIT + 6 * CNN11_LENGTH_IN;
                        addr_sram_act_for_8 <= ADDR_DCNN1_SRAM_ACT_INIT + 7 * CNN11_LENGTH_IN;
                        addr_sram_act_for_9 <= ADDR_DCNN1_SRAM_ACT_INIT + 8 * CNN11_LENGTH_IN;
                        addr_sram_act_for_10 <= ADDR_DCNN1_SRAM_ACT_INIT + 9 * CNN11_LENGTH_IN;
                        addr_sram_act_for_11 <= ADDR_DCNN1_SRAM_ACT_INIT + 10 * CNN11_LENGTH_IN;
                        addr_sram_act_for_12 <= ADDR_DCNN1_SRAM_ACT_INIT + 11 * CNN11_LENGTH_IN;
                        addr_sram_act_for_13 <= ADDR_DCNN1_SRAM_ACT_INIT + 12 * CNN11_LENGTH_IN;
                        addr_sram_act_for_14 <= ADDR_DCNN1_SRAM_ACT_INIT + 13 * CNN11_LENGTH_IN;
                        addr_sram_act_for_15 <= ADDR_DCNN1_SRAM_ACT_INIT + 14 * CNN11_LENGTH_IN;
                        addr_sram_act_for_16 <= ADDR_DCNN1_SRAM_ACT_INIT + 15 * CNN11_LENGTH_IN;
                        addr_sram_act_for_17 <= ADDR_DCNN1_SRAM_ACT_INIT + 16 * CNN11_LENGTH_IN;
                        addr_sram_act_for_18 <= ADDR_DCNN1_SRAM_ACT_INIT + 17 * CNN11_LENGTH_IN;
                        addr_sram_act_for_19 <= ADDR_DCNN1_SRAM_ACT_INIT + 18 * CNN11_LENGTH_IN;
                        addr_sram_act_for_20 <= ADDR_DCNN1_SRAM_ACT_INIT + 19 * CNN11_LENGTH_IN;
                        addr_sram_act_for_21 <= ADDR_DCNN1_SRAM_ACT_INIT + 20 * CNN11_LENGTH_IN;
                        addr_sram_act_for_22 <= ADDR_DCNN1_SRAM_ACT_INIT + 21 * CNN11_LENGTH_IN;
                        addr_sram_act_for_23 <= ADDR_DCNN1_SRAM_ACT_INIT + 22 * CNN11_LENGTH_IN;
                        addr_sram_act_for_24 <= ADDR_DCNN1_SRAM_ACT_INIT + 23 * CNN11_LENGTH_IN;
                        addr_sram_act_for_25 <= ADDR_DCNN1_SRAM_ACT_INIT + 24 * CNN11_LENGTH_IN;
                        addr_sram_act_for_26 <= ADDR_DCNN1_SRAM_ACT_INIT + 25 * CNN11_LENGTH_IN;
                        addr_sram_act_for_27 <= ADDR_DCNN1_SRAM_ACT_INIT + 26 * CNN11_LENGTH_IN;
                        addr_sram_act_for_28 <= ADDR_DCNN1_SRAM_ACT_INIT + 27 * CNN11_LENGTH_IN;
                        addr_sram_act_for_29 <= ADDR_DCNN1_SRAM_ACT_INIT + 28 * CNN11_LENGTH_IN;
                        addr_sram_act_for_30 <= ADDR_DCNN1_SRAM_ACT_INIT + 29 * CNN11_LENGTH_IN;
                        addr_sram_act_for_31 <= ADDR_DCNN1_SRAM_ACT_INIT + 30 * CNN11_LENGTH_IN;
                        addr_sram_act_for_32 <= ADDR_DCNN1_SRAM_ACT_INIT + 31 * CNN11_LENGTH_IN;                    
                    end
                    else ;

                end
                // if (layer_done) begin
                //     addr_sram_act_for_1 <= ADDR_DCNN1_SRAM_ACT_INIT;
                //     addr_sram_act_for_2 <= ADDR_DCNN1_SRAM_ACT_INIT + CNN11_LENGTH_IN;
                //     addr_sram_act_for_3 <= ADDR_DCNN1_SRAM_ACT_INIT + 2 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_4 <= ADDR_DCNN1_SRAM_ACT_INIT + 3 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_5 <= ADDR_DCNN1_SRAM_ACT_INIT + 4 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_6 <= ADDR_DCNN1_SRAM_ACT_INIT + 5 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_7 <= ADDR_DCNN1_SRAM_ACT_INIT + 6 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_8 <= ADDR_DCNN1_SRAM_ACT_INIT + 7 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_9 <= ADDR_DCNN1_SRAM_ACT_INIT + 8 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_10 <= ADDR_DCNN1_SRAM_ACT_INIT + 9 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_11 <= ADDR_DCNN1_SRAM_ACT_INIT + 10 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_12 <= ADDR_DCNN1_SRAM_ACT_INIT + 11 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_13 <= ADDR_DCNN1_SRAM_ACT_INIT + 12 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_14 <= ADDR_DCNN1_SRAM_ACT_INIT + 13 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_15 <= ADDR_DCNN1_SRAM_ACT_INIT + 14 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_16 <= ADDR_DCNN1_SRAM_ACT_INIT + 15 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_17 <= ADDR_DCNN1_SRAM_ACT_INIT + 16 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_18 <= ADDR_DCNN1_SRAM_ACT_INIT + 17 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_19 <= ADDR_DCNN1_SRAM_ACT_INIT + 18 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_20 <= ADDR_DCNN1_SRAM_ACT_INIT + 19 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_21 <= ADDR_DCNN1_SRAM_ACT_INIT + 20 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_22 <= ADDR_DCNN1_SRAM_ACT_INIT + 21 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_23 <= ADDR_DCNN1_SRAM_ACT_INIT + 22 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_24 <= ADDR_DCNN1_SRAM_ACT_INIT + 23 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_25 <= ADDR_DCNN1_SRAM_ACT_INIT + 24 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_26 <= ADDR_DCNN1_SRAM_ACT_INIT + 25 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_27 <= ADDR_DCNN1_SRAM_ACT_INIT + 26 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_28 <= ADDR_DCNN1_SRAM_ACT_INIT + 27 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_29 <= ADDR_DCNN1_SRAM_ACT_INIT + 28 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_30 <= ADDR_DCNN1_SRAM_ACT_INIT + 29 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_31 <= ADDR_DCNN1_SRAM_ACT_INIT + 30 * CNN11_LENGTH_IN;
                //     addr_sram_act_for_32 <= ADDR_DCNN1_SRAM_ACT_INIT + 31 * CNN11_LENGTH_IN;                    
                // end
            end
            else if (decoder_top_state_cnn ) begin
                if (sram_act_we) begin
                    cnt_re_sram  <= cnt_re_sram;  
                end
                else begin
                    if ((conv_state == load_a) & !addr_sram_act_re_end_1) begin                    
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_1 <= 0;
                            case (padding_crl)
                                0: begin
                                    if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_1 <= addr_sram_act_for_1 + 1;
                                    else addr_sram_act_for_1 <= addr_sram_act_for_1;                                    
                                end
                                1: begin
                                    if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_1 <= addr_sram_act_for_1 + 1;
                                    else addr_sram_act_for_1 <= addr_sram_act_for_1;
                                end
                                2:begin
                                    if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_1 <= addr_sram_act_for_1 + 1;
                                    else addr_sram_act_for_1 <= addr_sram_act_for_1;                           
                                end
                                3:begin
                                    if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_1 <= addr_sram_act_for_1 + 1;
                                    else addr_sram_act_for_1 <= addr_sram_act_for_1;                                  
                                end
                                4:begin
                                    if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_1 <= addr_sram_act_for_1 + 1;
                                    else addr_sram_act_for_1 <= addr_sram_act_for_1;                             
                                end
                            endcase
                        end  
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_1 <= 1;
                            if (decoder_top_state == cnn11) begin
                                if (padding_crl == 0) addr_sram_act_for_1 <=  addr_sram_act_for_1 - 3;
                                else if ((padding_crl == 1)  | (padding_crl == 2) ) addr_sram_act_for_1 <= ADDR_DCNN1_SRAM_ACT_INIT + 0*CNN11_LENGTH_IN;
                                else if (padding_crl == 3) addr_sram_act_for_1 <= ADDR_DCNN1_SRAM_ACT_INIT + 125 + 0*CNN11_LENGTH_IN;
                                else  addr_sram_act_for_1 <= ADDR_DCNN1_SRAM_ACT_INIT + 0*CNN11_LENGTH_IN ; 

                            end
                            else if (decoder_top_state == cnn12) begin
                                if (cnt_cho < CNN11_CHOUT/2 ) begin
                                    if (padding_crl == 0) addr_sram_act_for_1 <=  addr_sram_act_for_1 - 3;
                                    else if ((padding_crl == 1)  | (padding_crl == 2) ) addr_sram_act_for_1 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 0*CNN12_LENGTH_IN;
                                    else if (padding_crl == 3) addr_sram_act_for_1 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 125 + 0*CNN12_LENGTH_IN;
                                    else addr_sram_act_for_1 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 0*CNN12_LENGTH_IN ;  
                                end
                                else begin
                                    if (padding_crl == 0) addr_sram_act_for_1 <=  addr_sram_act_for_1 - 3;
                                    else if ((padding_crl == 1)  | (padding_crl == 2) ) addr_sram_act_for_1 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 0*CNN12_LENGTH_IN;
                                    else if (padding_crl == 3) addr_sram_act_for_1 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125 + 0*CNN12_LENGTH_IN;
                                    else addr_sram_act_for_1 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 0*CNN12_LENGTH_IN ;                                      
                                end
                            end
                            else if (decoder_top_state == cnn21) begin
                                if (padding_crl == 0) addr_sram_act_for_1 <=  addr_sram_act_for_1 - 3;
                                else if ((padding_crl == 1)  | (padding_crl == 2) ) addr_sram_act_for_1 <= ADDR_DCNN2_SRAM_ACT_INIT + 0*CNN21_LENGTH_IN;
                                else if (padding_crl == 3) addr_sram_act_for_1 <= ADDR_DCNN2_SRAM_ACT_INIT + 253 + 0*CNN21_LENGTH_IN;
                                else  addr_sram_act_for_1 <= ADDR_DCNN2_SRAM_ACT_INIT + 0*CNN21_LENGTH_IN ;                                 
                            end
                            else begin
                                if (padding_crl == 0) addr_sram_act_for_1 <=  addr_sram_act_for_1 - 3;
                                else if ((padding_crl == 1)  | (padding_crl == 2) ) addr_sram_act_for_1 <= ADDR_CNN21_SRAM_ACT_INIT + 0*CNN22_LENGTH_IN;
                                else if (padding_crl == 3) addr_sram_act_for_1 <= ADDR_CNN21_SRAM_ACT_INIT + 253 + 0*CNN22_LENGTH_IN;
                                else  addr_sram_act_for_1 <= ADDR_CNN21_SRAM_ACT_INIT + 0*CNN22_LENGTH_IN ;                                 
                            end

                        end   
                    end
                     else if ((conv_state_mem[0] == load_a) & !addr_sram_act_re_end_2) begin
                            addr_sram_act_re_end_1 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_2 <= 0;
                                case (padding_crl_mem[0])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_2 <= addr_sram_act_for_2 + 1;
                                        else addr_sram_act_for_2 <= addr_sram_act_for_2;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_2 <= addr_sram_act_for_2 + 1;
                                        else addr_sram_act_for_2 <= addr_sram_act_for_2;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_2 <= addr_sram_act_for_2 + 1;
                                        else addr_sram_act_for_2 <= addr_sram_act_for_2;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_2 <= addr_sram_act_for_2 + 1;
                                        else addr_sram_act_for_2 <= addr_sram_act_for_2;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_2 <= addr_sram_act_for_2 + 1;
                                        else addr_sram_act_for_2 <= addr_sram_act_for_2;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_2 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[0] == 0) addr_sram_act_for_2 <=  addr_sram_act_for_2 - 3;
                                    else if ((padding_crl_mem[0] == 1)  | (padding_crl_mem[0] == 2) ) addr_sram_act_for_2 <= ADDR_DCNN1_SRAM_ACT_INIT + 1*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[0] == 3) addr_sram_act_for_2 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  1*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_2 <= ADDR_DCNN1_SRAM_ACT_INIT + 1*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[0] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[0] == 0) addr_sram_act_for_2 <=  addr_sram_act_for_2 - 3;
                                        else if ((padding_crl_mem[0] == 1)  | (padding_crl_mem[0] == 2) ) addr_sram_act_for_2 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 1*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[0] == 3) addr_sram_act_for_2 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  1*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_2 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 1*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[0] == 0) addr_sram_act_for_2 <=  addr_sram_act_for_2 - 3;
                                        else if ((padding_crl_mem[0] == 1)  | (padding_crl_mem[0] == 2) ) addr_sram_act_for_2 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 1*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[0] == 3) addr_sram_act_for_2 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  1*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_2 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 1*CNN12_LENGTH_IN ;
                                    end
                                end
                                else if (decoder_top_state == cnn21) begin
                                    if (padding_crl_mem[0] == 0) addr_sram_act_for_2 <=  addr_sram_act_for_2 - 3;
                                    else if ((padding_crl_mem[0] == 1)  | (padding_crl_mem[0] == 2) ) addr_sram_act_for_2 <= ADDR_DCNN2_SRAM_ACT_INIT + 1*CNN21_LENGTH_IN;
                                    else if (padding_crl_mem[0] == 3) addr_sram_act_for_2 <= ADDR_DCNN2_SRAM_ACT_INIT + 253 + 1*CNN21_LENGTH_IN;
                                    else  addr_sram_act_for_2 <= ADDR_DCNN2_SRAM_ACT_INIT + 1*CNN21_LENGTH_IN ; 
                                end
                                else begin
                                    if (padding_crl_mem[0] == 0) addr_sram_act_for_2 <=  addr_sram_act_for_2 - 3;
                                    else if ((padding_crl_mem[0] == 1)  | (padding_crl_mem[0] == 2) ) addr_sram_act_for_2 <= ADDR_CNN21_SRAM_ACT_INIT + 1*CNN22_LENGTH_IN;
                                    else if (padding_crl_mem[0] == 3) addr_sram_act_for_2 <= ADDR_CNN21_SRAM_ACT_INIT + 253 + 1*CNN22_LENGTH_IN;
                                    else  addr_sram_act_for_2 <= ADDR_CNN21_SRAM_ACT_INIT + 1*CNN22_LENGTH_IN ;                  
                                end
                            end   
                        end
                     else if ((conv_state_mem[1] == load_a) & !addr_sram_act_re_end_3) begin
                         addr_sram_act_re_end_2 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_3 <= 0;
                                case (padding_crl_mem[1])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_3 <= addr_sram_act_for_3 + 1;
                                        else addr_sram_act_for_3 <= addr_sram_act_for_3;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_3 <= addr_sram_act_for_3 + 1;
                                        else addr_sram_act_for_3 <= addr_sram_act_for_3;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_3 <= addr_sram_act_for_3 + 1;
                                        else addr_sram_act_for_3 <= addr_sram_act_for_3;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_3 <= addr_sram_act_for_3 + 1;
                                        else addr_sram_act_for_3 <= addr_sram_act_for_3;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_3 <= addr_sram_act_for_3 + 1;
                                        else addr_sram_act_for_3 <= addr_sram_act_for_3;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_3 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[1] == 0) addr_sram_act_for_3 <=  addr_sram_act_for_3 - 3;
                                    else if ((padding_crl_mem[1] == 1)  | (padding_crl_mem[1] == 2) ) addr_sram_act_for_3 <= ADDR_DCNN1_SRAM_ACT_INIT + 2*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[1] == 3) addr_sram_act_for_3 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  2*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_3 <= ADDR_DCNN1_SRAM_ACT_INIT + 2*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[1] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[1] == 0) addr_sram_act_for_3 <=  addr_sram_act_for_3 - 3;
                                        else if ((padding_crl_mem[1] == 1)  | (padding_crl_mem[1] == 2) ) addr_sram_act_for_3 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 2*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[1] == 3) addr_sram_act_for_3 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  2*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_3 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 2*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[1] == 0) addr_sram_act_for_3 <=  addr_sram_act_for_3 - 3;
                                        else if ((padding_crl_mem[1] == 1)  | (padding_crl_mem[1] == 2) ) addr_sram_act_for_3 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 2*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[1] == 3) addr_sram_act_for_3 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  2*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_3 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 2*CNN12_LENGTH_IN ;
                                    end
                                end
                                else if (decoder_top_state == cnn21) begin
                                    if (padding_crl_mem[1] == 0) addr_sram_act_for_3 <=  addr_sram_act_for_3 - 3;
                                    else if ((padding_crl_mem[1] == 1)  | (padding_crl_mem[1] == 2) ) addr_sram_act_for_3 <= ADDR_DCNN2_SRAM_ACT_INIT + 2*CNN21_LENGTH_IN;
                                    else if (padding_crl_mem[1] == 3) addr_sram_act_for_3 <= ADDR_DCNN2_SRAM_ACT_INIT + 253 + 2*CNN21_LENGTH_IN;
                                    else  addr_sram_act_for_3 <= ADDR_DCNN2_SRAM_ACT_INIT + 2*CNN21_LENGTH_IN ; 
                                end
                                else begin
                                    if (padding_crl_mem[1] == 0) addr_sram_act_for_3 <=  addr_sram_act_for_3 - 3;
                                    else if ((padding_crl_mem[1] == 1)  | (padding_crl_mem[1] == 2) ) addr_sram_act_for_3 <= ADDR_CNN21_SRAM_ACT_INIT + 2*CNN22_LENGTH_IN;
                                    else if (padding_crl_mem[1] == 3) addr_sram_act_for_3 <= ADDR_CNN21_SRAM_ACT_INIT + 253 + 2*CNN22_LENGTH_IN;
                                    else  addr_sram_act_for_3 <= ADDR_CNN21_SRAM_ACT_INIT + 2*CNN22_LENGTH_IN ;                  
                                end
                            end   
                        end
                     else if ((conv_state_mem[2] == load_a) & !addr_sram_act_re_end_4) begin
                         addr_sram_act_re_end_3 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_4 <= 0;
                                case (padding_crl_mem[2])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_4 <= addr_sram_act_for_4 + 1;
                                        else addr_sram_act_for_4 <= addr_sram_act_for_4;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_4 <= addr_sram_act_for_4 + 1;
                                        else addr_sram_act_for_4 <= addr_sram_act_for_4;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_4 <= addr_sram_act_for_4 + 1;
                                        else addr_sram_act_for_4 <= addr_sram_act_for_4;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_4 <= addr_sram_act_for_4 + 1;
                                        else addr_sram_act_for_4 <= addr_sram_act_for_4;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_4 <= addr_sram_act_for_4 + 1;
                                        else addr_sram_act_for_4 <= addr_sram_act_for_4;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_4 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[2] == 0) addr_sram_act_for_4 <=  addr_sram_act_for_4 - 3;
                                    else if ((padding_crl_mem[2] == 1)  | (padding_crl_mem[2] == 2) ) addr_sram_act_for_4 <= ADDR_DCNN1_SRAM_ACT_INIT + 3*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[2] == 3) addr_sram_act_for_4 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  3*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_4 <= ADDR_DCNN1_SRAM_ACT_INIT + 3*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[2] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[2] == 0) addr_sram_act_for_4 <=  addr_sram_act_for_4 - 3;
                                        else if ((padding_crl_mem[2] == 1)  | (padding_crl_mem[2] == 2) ) addr_sram_act_for_4 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 3*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[2] == 3) addr_sram_act_for_4 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  3*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_4 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 3*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[2] == 0) addr_sram_act_for_4 <=  addr_sram_act_for_4 - 3;
                                        else if ((padding_crl_mem[2] == 1)  | (padding_crl_mem[2] == 2) ) addr_sram_act_for_4 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 3*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[2] == 3) addr_sram_act_for_4 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  3*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_4 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 3*CNN12_LENGTH_IN ;
                                    end
                                end
                                else if (decoder_top_state == cnn21) begin
                                    if (padding_crl_mem[2] == 0) addr_sram_act_for_4 <=  addr_sram_act_for_4 - 3;
                                    else if ((padding_crl_mem[2] == 1)  | (padding_crl_mem[2] == 2) ) addr_sram_act_for_4 <= ADDR_DCNN2_SRAM_ACT_INIT + 3*CNN21_LENGTH_IN;
                                    else if (padding_crl_mem[2] == 3) addr_sram_act_for_4 <= ADDR_DCNN2_SRAM_ACT_INIT + 253 + 3*CNN21_LENGTH_IN;
                                    else  addr_sram_act_for_4 <= ADDR_DCNN2_SRAM_ACT_INIT + 3*CNN21_LENGTH_IN ; 
                                end
                                else begin
                                    if (padding_crl_mem[2] == 0) addr_sram_act_for_4 <=  addr_sram_act_for_4 - 3;
                                    else if ((padding_crl_mem[2] == 1)  | (padding_crl_mem[2] == 2) ) addr_sram_act_for_4 <= ADDR_CNN21_SRAM_ACT_INIT + 3*CNN22_LENGTH_IN;
                                    else if (padding_crl_mem[2] == 3) addr_sram_act_for_4 <= ADDR_CNN21_SRAM_ACT_INIT + 253 + 3*CNN22_LENGTH_IN;
                                    else  addr_sram_act_for_4 <= ADDR_CNN21_SRAM_ACT_INIT + 3*CNN22_LENGTH_IN ;                  
                                end
                            end   
                        end
                     else if ((conv_state_mem[3] == load_a) & !addr_sram_act_re_end_5) begin
                         addr_sram_act_re_end_4 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_5 <= 0;
                                case (padding_crl_mem[3])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_5 <= addr_sram_act_for_5 + 1;
                                        else addr_sram_act_for_5 <= addr_sram_act_for_5;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_5 <= addr_sram_act_for_5 + 1;
                                        else addr_sram_act_for_5 <= addr_sram_act_for_5;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_5 <= addr_sram_act_for_5 + 1;
                                        else addr_sram_act_for_5 <= addr_sram_act_for_5;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_5 <= addr_sram_act_for_5 + 1;
                                        else addr_sram_act_for_5 <= addr_sram_act_for_5;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_5 <= addr_sram_act_for_5 + 1;
                                        else addr_sram_act_for_5 <= addr_sram_act_for_5;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_5 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[3] == 0) addr_sram_act_for_5 <=  addr_sram_act_for_5 - 3;
                                    else if ((padding_crl_mem[3] == 1)  | (padding_crl_mem[3] == 2) ) addr_sram_act_for_5 <= ADDR_DCNN1_SRAM_ACT_INIT + 4*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[3] == 3) addr_sram_act_for_5 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  4*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_5 <= ADDR_DCNN1_SRAM_ACT_INIT + 4*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[3] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[3] == 0) addr_sram_act_for_5 <=  addr_sram_act_for_5 - 3;
                                        else if ((padding_crl_mem[3] == 1)  | (padding_crl_mem[3] == 2) ) addr_sram_act_for_5 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 4*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[3] == 3) addr_sram_act_for_5 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  4*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_5 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 4*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[3] == 0) addr_sram_act_for_5 <=  addr_sram_act_for_5 - 3;
                                        else if ((padding_crl_mem[3] == 1)  | (padding_crl_mem[3] == 2) ) addr_sram_act_for_5 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 4*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[3] == 3) addr_sram_act_for_5 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  4*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_5 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 4*CNN12_LENGTH_IN ;
                                    end
                                end
                                else if (decoder_top_state == cnn21) begin
                                    if (padding_crl_mem[3] == 0) addr_sram_act_for_5 <=  addr_sram_act_for_5 - 3;
                                    else if ((padding_crl_mem[3] == 1)  | (padding_crl_mem[3] == 2) ) addr_sram_act_for_5 <= ADDR_DCNN2_SRAM_ACT_INIT + 4*CNN21_LENGTH_IN;
                                    else if (padding_crl_mem[3] == 3) addr_sram_act_for_5 <= ADDR_DCNN2_SRAM_ACT_INIT + 253 + 4*CNN21_LENGTH_IN;
                                    else  addr_sram_act_for_5 <= ADDR_DCNN2_SRAM_ACT_INIT + 4*CNN21_LENGTH_IN ; 
                                end
                                else begin
                                    if (padding_crl_mem[3] == 0) addr_sram_act_for_5 <=  addr_sram_act_for_5 - 3;
                                    else if ((padding_crl_mem[3] == 1)  | (padding_crl_mem[3] == 2) ) addr_sram_act_for_5 <= ADDR_CNN21_SRAM_ACT_INIT + 4*CNN22_LENGTH_IN;
                                    else if (padding_crl_mem[3] == 3) addr_sram_act_for_5 <= ADDR_CNN21_SRAM_ACT_INIT + 253 + 4*CNN22_LENGTH_IN;
                                    else  addr_sram_act_for_5 <= ADDR_CNN21_SRAM_ACT_INIT + 4*CNN22_LENGTH_IN ;                  
                                end
                            end   
                        end
                     else if ((conv_state_mem[4] == load_a) & !addr_sram_act_re_end_6) begin
                         addr_sram_act_re_end_5 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_6 <= 0;
                                case (padding_crl_mem[4])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_6 <= addr_sram_act_for_6 + 1;
                                        else addr_sram_act_for_6 <= addr_sram_act_for_6;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_6 <= addr_sram_act_for_6 + 1;
                                        else addr_sram_act_for_6 <= addr_sram_act_for_6;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_6 <= addr_sram_act_for_6 + 1;
                                        else addr_sram_act_for_6 <= addr_sram_act_for_6;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_6 <= addr_sram_act_for_6 + 1;
                                        else addr_sram_act_for_6 <= addr_sram_act_for_6;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_6 <= addr_sram_act_for_6 + 1;
                                        else addr_sram_act_for_6 <= addr_sram_act_for_6;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_6 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[4] == 0) addr_sram_act_for_6 <=  addr_sram_act_for_6 - 3;
                                    else if ((padding_crl_mem[4] == 1)  | (padding_crl_mem[4] == 2) ) addr_sram_act_for_6 <= ADDR_DCNN1_SRAM_ACT_INIT + 5*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[4] == 3) addr_sram_act_for_6 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  5*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_6 <= ADDR_DCNN1_SRAM_ACT_INIT + 5*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[4] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[4] == 0) addr_sram_act_for_6 <=  addr_sram_act_for_6 - 3;
                                        else if ((padding_crl_mem[4] == 1)  | (padding_crl_mem[4] == 2) ) addr_sram_act_for_6 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 5*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[4] == 3) addr_sram_act_for_6 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  5*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_6 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 5*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[4] == 0) addr_sram_act_for_6 <=  addr_sram_act_for_6 - 3;
                                        else if ((padding_crl_mem[4] == 1)  | (padding_crl_mem[4] == 2) ) addr_sram_act_for_6 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 5*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[4] == 3) addr_sram_act_for_6 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  5*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_6 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 5*CNN12_LENGTH_IN ;
                                    end
                                end
                                else if (decoder_top_state == cnn21) begin
                                    if (padding_crl_mem[4] == 0) addr_sram_act_for_6 <=  addr_sram_act_for_6 - 3;
                                    else if ((padding_crl_mem[4] == 1)  | (padding_crl_mem[4] == 2) ) addr_sram_act_for_6 <= ADDR_DCNN2_SRAM_ACT_INIT + 5*CNN21_LENGTH_IN;
                                    else if (padding_crl_mem[4] == 3) addr_sram_act_for_6 <= ADDR_DCNN2_SRAM_ACT_INIT + 253 + 5*CNN21_LENGTH_IN;
                                    else  addr_sram_act_for_6 <= ADDR_DCNN2_SRAM_ACT_INIT + 5*CNN21_LENGTH_IN ; 
                                end
                                else begin
                                    if (padding_crl_mem[4] == 0) addr_sram_act_for_6 <=  addr_sram_act_for_6 - 3;
                                    else if ((padding_crl_mem[4] == 1)  | (padding_crl_mem[4] == 2) ) addr_sram_act_for_6 <= ADDR_CNN21_SRAM_ACT_INIT + 5*CNN22_LENGTH_IN;
                                    else if (padding_crl_mem[4] == 3) addr_sram_act_for_6 <= ADDR_CNN21_SRAM_ACT_INIT + 253 + 5*CNN22_LENGTH_IN;
                                    else  addr_sram_act_for_6 <= ADDR_CNN21_SRAM_ACT_INIT + 5*CNN22_LENGTH_IN ;                  
                                end
                            end   
                        end
                     else if ((conv_state_mem[5] == load_a) & !addr_sram_act_re_end_7) begin
                         addr_sram_act_re_end_6 <= 0;
                         addr_sram_act_re_end_8 <= (decoder_top_state_8)? 0:addr_sram_act_re_end_8;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_7 <= 0;
                                case (padding_crl_mem[5])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_7 <= addr_sram_act_for_7 + 1;
                                        else addr_sram_act_for_7 <= addr_sram_act_for_7;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_7 <= addr_sram_act_for_7 + 1;
                                        else addr_sram_act_for_7 <= addr_sram_act_for_7;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_7 <= addr_sram_act_for_7 + 1;
                                        else addr_sram_act_for_7 <= addr_sram_act_for_7;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_7 <= addr_sram_act_for_7 + 1;
                                        else addr_sram_act_for_7 <= addr_sram_act_for_7;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_7 <= addr_sram_act_for_7 + 1;
                                        else addr_sram_act_for_7 <= addr_sram_act_for_7;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_7 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[5] == 0) addr_sram_act_for_7 <=  addr_sram_act_for_7 - 3;
                                    else if ((padding_crl_mem[5] == 1)  | (padding_crl_mem[5] == 2) ) addr_sram_act_for_7 <= ADDR_DCNN1_SRAM_ACT_INIT + 6*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[5] == 3) addr_sram_act_for_7 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  6*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_7 <= ADDR_DCNN1_SRAM_ACT_INIT + 6*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[5] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[5] == 0) addr_sram_act_for_7 <=  addr_sram_act_for_7 - 3;
                                        else if ((padding_crl_mem[5] == 1)  | (padding_crl_mem[5] == 2) ) addr_sram_act_for_7 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 6*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[5] == 3) addr_sram_act_for_7 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  6*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_7 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 6*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[5] == 0) addr_sram_act_for_7 <=  addr_sram_act_for_7 - 3;
                                        else if ((padding_crl_mem[5] == 1)  | (padding_crl_mem[5] == 2) ) addr_sram_act_for_7 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 6*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[5] == 3) addr_sram_act_for_7 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  6*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_7 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 6*CNN12_LENGTH_IN ;
                                    end
                                end
                                else if (decoder_top_state == cnn21) begin
                                    if (padding_crl_mem[5] == 0) addr_sram_act_for_7 <=  addr_sram_act_for_7 - 3;
                                    else if ((padding_crl_mem[5] == 1)  | (padding_crl_mem[5] == 2) ) addr_sram_act_for_7 <= ADDR_DCNN2_SRAM_ACT_INIT + 6*CNN21_LENGTH_IN;
                                    else if (padding_crl_mem[5] == 3) addr_sram_act_for_7 <= ADDR_DCNN2_SRAM_ACT_INIT + 253 + 6*CNN21_LENGTH_IN;
                                    else  addr_sram_act_for_7 <= ADDR_DCNN2_SRAM_ACT_INIT + 6*CNN21_LENGTH_IN ; 
                                end
                                else begin
                                    if (padding_crl_mem[5] == 0) addr_sram_act_for_7 <=  addr_sram_act_for_7 - 3;
                                    else if ((padding_crl_mem[5] == 1)  | (padding_crl_mem[5] == 2) ) addr_sram_act_for_7 <= ADDR_CNN21_SRAM_ACT_INIT + 6*CNN22_LENGTH_IN;
                                    else if (padding_crl_mem[5] == 3) addr_sram_act_for_7 <= ADDR_CNN21_SRAM_ACT_INIT + 253 + 6*CNN22_LENGTH_IN;
                                    else  addr_sram_act_for_7 <= ADDR_CNN21_SRAM_ACT_INIT + 6*CNN22_LENGTH_IN ;                  
                                end
                            end   
                        end
                     else if ((conv_state_mem[6] == load_a) & !addr_sram_act_re_end_8) begin
                         addr_sram_act_re_end_7 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_8 <= 0;
                                case (padding_crl_mem[6])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_8 <= addr_sram_act_for_8 + 1;
                                        else addr_sram_act_for_8 <= addr_sram_act_for_8;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_8 <= addr_sram_act_for_8 + 1;
                                        else addr_sram_act_for_8 <= addr_sram_act_for_8;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_8 <= addr_sram_act_for_8 + 1;
                                        else addr_sram_act_for_8 <= addr_sram_act_for_8;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_8 <= addr_sram_act_for_8 + 1;
                                        else addr_sram_act_for_8 <= addr_sram_act_for_8;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_8 <= addr_sram_act_for_8 + 1;
                                        else addr_sram_act_for_8 <= addr_sram_act_for_8;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_8 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[6] == 0) addr_sram_act_for_8 <=  addr_sram_act_for_8 - 3;
                                    else if ((padding_crl_mem[6] == 1)  | (padding_crl_mem[6] == 2) ) addr_sram_act_for_8 <= ADDR_DCNN1_SRAM_ACT_INIT + 7*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[6] == 3) addr_sram_act_for_8 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  7*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_8 <= ADDR_DCNN1_SRAM_ACT_INIT + 7*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[6] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[6] == 0) addr_sram_act_for_8 <=  addr_sram_act_for_8 - 3;
                                        else if ((padding_crl_mem[6] == 1)  | (padding_crl_mem[6] == 2) ) addr_sram_act_for_8 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 7*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[6] == 3) addr_sram_act_for_8 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  7*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_8 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 7*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[6] == 0) addr_sram_act_for_8 <=  addr_sram_act_for_8 - 3;
                                        else if ((padding_crl_mem[6] == 1)  | (padding_crl_mem[6] == 2) ) addr_sram_act_for_8 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 7*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[6] == 3) addr_sram_act_for_8 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  7*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_8 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 7*CNN12_LENGTH_IN ;
                                    end
                                end
                                else if (decoder_top_state == cnn21) begin
                                    if (padding_crl_mem[6] == 0) addr_sram_act_for_8 <=  addr_sram_act_for_8 - 3;
                                    else if ((padding_crl_mem[6] == 1)  | (padding_crl_mem[6] == 2) ) addr_sram_act_for_8 <= ADDR_DCNN2_SRAM_ACT_INIT + 7*CNN21_LENGTH_IN;
                                    else if (padding_crl_mem[6] == 3) addr_sram_act_for_8 <= ADDR_DCNN2_SRAM_ACT_INIT + 253 + 7*CNN21_LENGTH_IN;
                                    else  addr_sram_act_for_8 <= ADDR_DCNN2_SRAM_ACT_INIT + 7*CNN21_LENGTH_IN ; 
                                end
                                else begin
                                    if (padding_crl_mem[6] == 0) addr_sram_act_for_8 <=  addr_sram_act_for_8 - 3;
                                    else if ((padding_crl_mem[6] == 1)  | (padding_crl_mem[6] == 2) ) addr_sram_act_for_8 <= ADDR_CNN21_SRAM_ACT_INIT + 7*CNN22_LENGTH_IN;
                                    else if (padding_crl_mem[6] == 3) addr_sram_act_for_8 <= ADDR_CNN21_SRAM_ACT_INIT + 253 + 7*CNN22_LENGTH_IN;
                                    else  addr_sram_act_for_8 <= ADDR_CNN21_SRAM_ACT_INIT + 7*CNN22_LENGTH_IN ;                  
                                end
                            end   
                        end
                     else if ((conv_state_mem[7] == load_a) & !addr_sram_act_re_end_9) begin
                         addr_sram_act_re_end_8 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_9 <= 0;
                                case (padding_crl_mem[7])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_9 <= addr_sram_act_for_9 + 1;
                                        else addr_sram_act_for_9 <= addr_sram_act_for_9;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_9 <= addr_sram_act_for_9 + 1;
                                        else addr_sram_act_for_9 <= addr_sram_act_for_9;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_9 <= addr_sram_act_for_9 + 1;
                                        else addr_sram_act_for_9 <= addr_sram_act_for_9;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_9 <= addr_sram_act_for_9 + 1;
                                        else addr_sram_act_for_9 <= addr_sram_act_for_9;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_9 <= addr_sram_act_for_9 + 1;
                                        else addr_sram_act_for_9 <= addr_sram_act_for_9;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_9 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[7] == 0) addr_sram_act_for_9 <=  addr_sram_act_for_9 - 3;
                                    else if ((padding_crl_mem[7] == 1)  | (padding_crl_mem[7] == 2) ) addr_sram_act_for_9 <= ADDR_DCNN1_SRAM_ACT_INIT + 8*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[7] == 3) addr_sram_act_for_9 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  8*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_9 <= ADDR_DCNN1_SRAM_ACT_INIT + 8*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[7] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[7] == 0) addr_sram_act_for_9 <=  addr_sram_act_for_9 - 3;
                                        else if ((padding_crl_mem[7] == 1)  | (padding_crl_mem[7] == 2) ) addr_sram_act_for_9 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 8*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[7] == 3) addr_sram_act_for_9 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  8*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_9 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 8*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[7] == 0) addr_sram_act_for_9 <=  addr_sram_act_for_9 - 3;
                                        else if ((padding_crl_mem[7] == 1)  | (padding_crl_mem[7] == 2) ) addr_sram_act_for_9 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 8*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[7] == 3) addr_sram_act_for_9 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  8*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_9 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 8*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[8] == load_a) & !addr_sram_act_re_end_10) begin
                         addr_sram_act_re_end_9 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_10 <= 0;
                                case (padding_crl_mem[8])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_10 <= addr_sram_act_for_10 + 1;
                                        else addr_sram_act_for_10 <= addr_sram_act_for_10;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_10 <= addr_sram_act_for_10 + 1;
                                        else addr_sram_act_for_10 <= addr_sram_act_for_10;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_10 <= addr_sram_act_for_10 + 1;
                                        else addr_sram_act_for_10 <= addr_sram_act_for_10;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_10 <= addr_sram_act_for_10 + 1;
                                        else addr_sram_act_for_10 <= addr_sram_act_for_10;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_10 <= addr_sram_act_for_10 + 1;
                                        else addr_sram_act_for_10 <= addr_sram_act_for_10;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_10 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[8] == 0) addr_sram_act_for_10 <=  addr_sram_act_for_10 - 3;
                                    else if ((padding_crl_mem[8] == 1)  | (padding_crl_mem[8] == 2) ) addr_sram_act_for_10 <= ADDR_DCNN1_SRAM_ACT_INIT + 9*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[8] == 3) addr_sram_act_for_10 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  9*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_10 <= ADDR_DCNN1_SRAM_ACT_INIT + 9*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[8] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[8] == 0) addr_sram_act_for_10 <=  addr_sram_act_for_10 - 3;
                                        else if ((padding_crl_mem[8] == 1)  | (padding_crl_mem[8] == 2) ) addr_sram_act_for_10 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 9*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[8] == 3) addr_sram_act_for_10 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  9*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_10 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 9*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[8] == 0) addr_sram_act_for_10 <=  addr_sram_act_for_10 - 3;
                                        else if ((padding_crl_mem[8] == 1)  | (padding_crl_mem[8] == 2) ) addr_sram_act_for_10 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 9*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[8] == 3) addr_sram_act_for_10 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  9*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_10 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 9*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[9] == load_a) & !addr_sram_act_re_end_11) begin
                         addr_sram_act_re_end_10 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_11 <= 0;
                                case (padding_crl_mem[9])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_11 <= addr_sram_act_for_11 + 1;
                                        else addr_sram_act_for_11 <= addr_sram_act_for_11;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_11 <= addr_sram_act_for_11 + 1;
                                        else addr_sram_act_for_11 <= addr_sram_act_for_11;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_11 <= addr_sram_act_for_11 + 1;
                                        else addr_sram_act_for_11 <= addr_sram_act_for_11;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_11 <= addr_sram_act_for_11 + 1;
                                        else addr_sram_act_for_11 <= addr_sram_act_for_11;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_11 <= addr_sram_act_for_11 + 1;
                                        else addr_sram_act_for_11 <= addr_sram_act_for_11;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_11 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[9] == 0) addr_sram_act_for_11 <=  addr_sram_act_for_11 - 3;
                                    else if ((padding_crl_mem[9] == 1)  | (padding_crl_mem[9] == 2) ) addr_sram_act_for_11 <= ADDR_DCNN1_SRAM_ACT_INIT + 10*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[9] == 3) addr_sram_act_for_11 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  10*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_11 <= ADDR_DCNN1_SRAM_ACT_INIT + 10*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[9] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[9] == 0) addr_sram_act_for_11 <=  addr_sram_act_for_11 - 3;
                                        else if ((padding_crl_mem[9] == 1)  | (padding_crl_mem[9] == 2) ) addr_sram_act_for_11 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 10*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[9] == 3) addr_sram_act_for_11 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  10*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_11 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 10*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[9] == 0) addr_sram_act_for_11 <=  addr_sram_act_for_11 - 3;
                                        else if ((padding_crl_mem[9] == 1)  | (padding_crl_mem[9] == 2) ) addr_sram_act_for_11 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 10*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[9] == 3) addr_sram_act_for_11 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  10*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_11 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 10*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[10] == load_a) & !addr_sram_act_re_end_12) begin
                         addr_sram_act_re_end_11 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_12 <= 0;
                                case (padding_crl_mem[10])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_12 <= addr_sram_act_for_12 + 1;
                                        else addr_sram_act_for_12 <= addr_sram_act_for_12;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_12 <= addr_sram_act_for_12 + 1;
                                        else addr_sram_act_for_12 <= addr_sram_act_for_12;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_12 <= addr_sram_act_for_12 + 1;
                                        else addr_sram_act_for_12 <= addr_sram_act_for_12;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_12 <= addr_sram_act_for_12 + 1;
                                        else addr_sram_act_for_12 <= addr_sram_act_for_12;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_12 <= addr_sram_act_for_12 + 1;
                                        else addr_sram_act_for_12 <= addr_sram_act_for_12;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_12 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[10] == 0) addr_sram_act_for_12 <=  addr_sram_act_for_12 - 3;
                                    else if ((padding_crl_mem[10] == 1)  | (padding_crl_mem[10] == 2) ) addr_sram_act_for_12 <= ADDR_DCNN1_SRAM_ACT_INIT + 11*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[10] == 3) addr_sram_act_for_12 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  11*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_12 <= ADDR_DCNN1_SRAM_ACT_INIT + 11*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[10] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[10] == 0) addr_sram_act_for_12 <=  addr_sram_act_for_12 - 3;
                                        else if ((padding_crl_mem[10] == 1)  | (padding_crl_mem[10] == 2) ) addr_sram_act_for_12 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 11*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[10] == 3) addr_sram_act_for_12 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  11*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_12 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 11*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[10] == 0) addr_sram_act_for_12 <=  addr_sram_act_for_12 - 3;
                                        else if ((padding_crl_mem[10] == 1)  | (padding_crl_mem[10] == 2) ) addr_sram_act_for_12 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 11*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[10] == 3) addr_sram_act_for_12 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  11*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_12 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 11*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[11] == load_a) & !addr_sram_act_re_end_13) begin
                         addr_sram_act_re_end_12 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_13 <= 0;
                                case (padding_crl_mem[11])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_13 <= addr_sram_act_for_13 + 1;
                                        else addr_sram_act_for_13 <= addr_sram_act_for_13;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_13 <= addr_sram_act_for_13 + 1;
                                        else addr_sram_act_for_13 <= addr_sram_act_for_13;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_13 <= addr_sram_act_for_13 + 1;
                                        else addr_sram_act_for_13 <= addr_sram_act_for_13;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_13 <= addr_sram_act_for_13 + 1;
                                        else addr_sram_act_for_13 <= addr_sram_act_for_13;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_13 <= addr_sram_act_for_13 + 1;
                                        else addr_sram_act_for_13 <= addr_sram_act_for_13;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_13 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[11] == 0) addr_sram_act_for_13 <=  addr_sram_act_for_13 - 3;
                                    else if ((padding_crl_mem[11] == 1)  | (padding_crl_mem[11] == 2) ) addr_sram_act_for_13 <= ADDR_DCNN1_SRAM_ACT_INIT + 12*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[11] == 3) addr_sram_act_for_13 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  12*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_13 <= ADDR_DCNN1_SRAM_ACT_INIT + 12*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[11] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[11] == 0) addr_sram_act_for_13 <=  addr_sram_act_for_13 - 3;
                                        else if ((padding_crl_mem[11] == 1)  | (padding_crl_mem[11] == 2) ) addr_sram_act_for_13 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 12*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[11] == 3) addr_sram_act_for_13 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  12*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_13 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 12*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[11] == 0) addr_sram_act_for_13 <=  addr_sram_act_for_13 - 3;
                                        else if ((padding_crl_mem[11] == 1)  | (padding_crl_mem[11] == 2) ) addr_sram_act_for_13 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 12*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[11] == 3) addr_sram_act_for_13 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  12*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_13 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 12*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[12] == load_a) & !addr_sram_act_re_end_14) begin
                         addr_sram_act_re_end_13 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_14 <= 0;
                                case (padding_crl_mem[12])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_14 <= addr_sram_act_for_14 + 1;
                                        else addr_sram_act_for_14 <= addr_sram_act_for_14;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_14 <= addr_sram_act_for_14 + 1;
                                        else addr_sram_act_for_14 <= addr_sram_act_for_14;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_14 <= addr_sram_act_for_14 + 1;
                                        else addr_sram_act_for_14 <= addr_sram_act_for_14;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_14 <= addr_sram_act_for_14 + 1;
                                        else addr_sram_act_for_14 <= addr_sram_act_for_14;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_14 <= addr_sram_act_for_14 + 1;
                                        else addr_sram_act_for_14 <= addr_sram_act_for_14;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_14 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[12] == 0) addr_sram_act_for_14 <=  addr_sram_act_for_14 - 3;
                                    else if ((padding_crl_mem[12] == 1)  | (padding_crl_mem[12] == 2) ) addr_sram_act_for_14 <= ADDR_DCNN1_SRAM_ACT_INIT + 13*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[12] == 3) addr_sram_act_for_14 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  13*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_14 <= ADDR_DCNN1_SRAM_ACT_INIT + 13*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[12] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[12] == 0) addr_sram_act_for_14 <=  addr_sram_act_for_14 - 3;
                                        else if ((padding_crl_mem[12] == 1)  | (padding_crl_mem[12] == 2) ) addr_sram_act_for_14 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 13*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[12] == 3) addr_sram_act_for_14 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  13*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_14 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 13*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[12] == 0) addr_sram_act_for_14 <=  addr_sram_act_for_14 - 3;
                                        else if ((padding_crl_mem[12] == 1)  | (padding_crl_mem[12] == 2) ) addr_sram_act_for_14 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 13*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[12] == 3) addr_sram_act_for_14 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  13*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_14 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 13*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[13] == load_a) & !addr_sram_act_re_end_15) begin
                         addr_sram_act_re_end_14 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_15 <= 0;
                                case (padding_crl_mem[13])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_15 <= addr_sram_act_for_15 + 1;
                                        else addr_sram_act_for_15 <= addr_sram_act_for_15;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_15 <= addr_sram_act_for_15 + 1;
                                        else addr_sram_act_for_15 <= addr_sram_act_for_15;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_15 <= addr_sram_act_for_15 + 1;
                                        else addr_sram_act_for_15 <= addr_sram_act_for_15;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_15 <= addr_sram_act_for_15 + 1;
                                        else addr_sram_act_for_15 <= addr_sram_act_for_15;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_15 <= addr_sram_act_for_15 + 1;
                                        else addr_sram_act_for_15 <= addr_sram_act_for_15;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_15 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[13] == 0) addr_sram_act_for_15 <=  addr_sram_act_for_15 - 3;
                                    else if ((padding_crl_mem[13] == 1)  | (padding_crl_mem[13] == 2) ) addr_sram_act_for_15 <= ADDR_DCNN1_SRAM_ACT_INIT + 14*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[13] == 3) addr_sram_act_for_15 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  14*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_15 <= ADDR_DCNN1_SRAM_ACT_INIT + 14*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[13] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[13] == 0) addr_sram_act_for_15 <=  addr_sram_act_for_15 - 3;
                                        else if ((padding_crl_mem[13] == 1)  | (padding_crl_mem[13] == 2) ) addr_sram_act_for_15 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 14*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[13] == 3) addr_sram_act_for_15 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  14*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_15 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 14*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[13] == 0) addr_sram_act_for_15 <=  addr_sram_act_for_15 - 3;
                                        else if ((padding_crl_mem[13] == 1)  | (padding_crl_mem[13] == 2) ) addr_sram_act_for_15 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 14*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[13] == 3) addr_sram_act_for_15 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  14*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_15 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 14*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[14] == load_a) & !addr_sram_act_re_end_16) begin
                         addr_sram_act_re_end_15 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_16 <= 0;
                                case (padding_crl_mem[14])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_16 <= addr_sram_act_for_16 + 1;
                                        else addr_sram_act_for_16 <= addr_sram_act_for_16;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_16 <= addr_sram_act_for_16 + 1;
                                        else addr_sram_act_for_16 <= addr_sram_act_for_16;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_16 <= addr_sram_act_for_16 + 1;
                                        else addr_sram_act_for_16 <= addr_sram_act_for_16;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_16 <= addr_sram_act_for_16 + 1;
                                        else addr_sram_act_for_16 <= addr_sram_act_for_16;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_16 <= addr_sram_act_for_16 + 1;
                                        else addr_sram_act_for_16 <= addr_sram_act_for_16;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_16 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[14] == 0) addr_sram_act_for_16 <=  addr_sram_act_for_16 - 3;
                                    else if ((padding_crl_mem[14] == 1)  | (padding_crl_mem[14] == 2) ) addr_sram_act_for_16 <= ADDR_DCNN1_SRAM_ACT_INIT + 15*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[14] == 3) addr_sram_act_for_16 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  15*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_16 <= ADDR_DCNN1_SRAM_ACT_INIT + 15*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[14] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[14] == 0) addr_sram_act_for_16 <=  addr_sram_act_for_16 - 3;
                                        else if ((padding_crl_mem[14] == 1)  | (padding_crl_mem[14] == 2) ) addr_sram_act_for_16 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 15*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[14] == 3) addr_sram_act_for_16 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  15*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_16 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 15*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[14] == 0) addr_sram_act_for_16 <=  addr_sram_act_for_16 - 3;
                                        else if ((padding_crl_mem[14] == 1)  | (padding_crl_mem[14] == 2) ) addr_sram_act_for_16 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 15*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[14] == 3) addr_sram_act_for_16 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  15*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_16 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 15*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[15] == load_a) & !addr_sram_act_re_end_17) begin
                         addr_sram_act_re_end_16 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_17 <= 0;
                                case (padding_crl_mem[15])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_17 <= addr_sram_act_for_17 + 1;
                                        else addr_sram_act_for_17 <= addr_sram_act_for_17;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_17 <= addr_sram_act_for_17 + 1;
                                        else addr_sram_act_for_17 <= addr_sram_act_for_17;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_17 <= addr_sram_act_for_17 + 1;
                                        else addr_sram_act_for_17 <= addr_sram_act_for_17;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_17 <= addr_sram_act_for_17 + 1;
                                        else addr_sram_act_for_17 <= addr_sram_act_for_17;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_17 <= addr_sram_act_for_17 + 1;
                                        else addr_sram_act_for_17 <= addr_sram_act_for_17;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_17 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[15] == 0) addr_sram_act_for_17 <=  addr_sram_act_for_17 - 3;
                                    else if ((padding_crl_mem[15] == 1)  | (padding_crl_mem[15] == 2) ) addr_sram_act_for_17 <= ADDR_DCNN1_SRAM_ACT_INIT + 16*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[15] == 3) addr_sram_act_for_17 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  16*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_17 <= ADDR_DCNN1_SRAM_ACT_INIT + 16*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[15] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[15] == 0) addr_sram_act_for_17 <=  addr_sram_act_for_17 - 3;
                                        else if ((padding_crl_mem[15] == 1)  | (padding_crl_mem[15] == 2) ) addr_sram_act_for_17 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 16*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[15] == 3) addr_sram_act_for_17 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  16*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_17 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 16*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[15] == 0) addr_sram_act_for_17 <=  addr_sram_act_for_17 - 3;
                                        else if ((padding_crl_mem[15] == 1)  | (padding_crl_mem[15] == 2) ) addr_sram_act_for_17 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 16*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[15] == 3) addr_sram_act_for_17 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  16*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_17 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 16*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[16] == load_a) & !addr_sram_act_re_end_18) begin
                         addr_sram_act_re_end_17 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_18 <= 0;
                                case (padding_crl_mem[16])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_18 <= addr_sram_act_for_18 + 1;
                                        else addr_sram_act_for_18 <= addr_sram_act_for_18;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_18 <= addr_sram_act_for_18 + 1;
                                        else addr_sram_act_for_18 <= addr_sram_act_for_18;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_18 <= addr_sram_act_for_18 + 1;
                                        else addr_sram_act_for_18 <= addr_sram_act_for_18;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_18 <= addr_sram_act_for_18 + 1;
                                        else addr_sram_act_for_18 <= addr_sram_act_for_18;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_18 <= addr_sram_act_for_18 + 1;
                                        else addr_sram_act_for_18 <= addr_sram_act_for_18;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_18 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[16] == 0) addr_sram_act_for_18 <=  addr_sram_act_for_18 - 3;
                                    else if ((padding_crl_mem[16] == 1)  | (padding_crl_mem[16] == 2) ) addr_sram_act_for_18 <= ADDR_DCNN1_SRAM_ACT_INIT + 17*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[16] == 3) addr_sram_act_for_18 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  17*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_18 <= ADDR_DCNN1_SRAM_ACT_INIT + 17*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[16] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[16] == 0) addr_sram_act_for_18 <=  addr_sram_act_for_18 - 3;
                                        else if ((padding_crl_mem[16] == 1)  | (padding_crl_mem[16] == 2) ) addr_sram_act_for_18 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 17*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[16] == 3) addr_sram_act_for_18 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  17*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_18 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 17*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[16] == 0) addr_sram_act_for_18 <=  addr_sram_act_for_18 - 3;
                                        else if ((padding_crl_mem[16] == 1)  | (padding_crl_mem[16] == 2) ) addr_sram_act_for_18 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 17*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[16] == 3) addr_sram_act_for_18 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  17*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_18 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 17*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[17] == load_a) & !addr_sram_act_re_end_19) begin
                         addr_sram_act_re_end_18 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_19 <= 0;
                                case (padding_crl_mem[17])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_19 <= addr_sram_act_for_19 + 1;
                                        else addr_sram_act_for_19 <= addr_sram_act_for_19;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_19 <= addr_sram_act_for_19 + 1;
                                        else addr_sram_act_for_19 <= addr_sram_act_for_19;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_19 <= addr_sram_act_for_19 + 1;
                                        else addr_sram_act_for_19 <= addr_sram_act_for_19;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_19 <= addr_sram_act_for_19 + 1;
                                        else addr_sram_act_for_19 <= addr_sram_act_for_19;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_19 <= addr_sram_act_for_19 + 1;
                                        else addr_sram_act_for_19 <= addr_sram_act_for_19;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_19 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[17] == 0) addr_sram_act_for_19 <=  addr_sram_act_for_19 - 3;
                                    else if ((padding_crl_mem[17] == 1)  | (padding_crl_mem[17] == 2) ) addr_sram_act_for_19 <= ADDR_DCNN1_SRAM_ACT_INIT + 18*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[17] == 3) addr_sram_act_for_19 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  18*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_19 <= ADDR_DCNN1_SRAM_ACT_INIT + 18*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[17] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[17] == 0) addr_sram_act_for_19 <=  addr_sram_act_for_19 - 3;
                                        else if ((padding_crl_mem[17] == 1)  | (padding_crl_mem[17] == 2) ) addr_sram_act_for_19 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 18*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[17] == 3) addr_sram_act_for_19 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  18*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_19 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 18*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[17] == 0) addr_sram_act_for_19 <=  addr_sram_act_for_19 - 3;
                                        else if ((padding_crl_mem[17] == 1)  | (padding_crl_mem[17] == 2) ) addr_sram_act_for_19 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 18*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[17] == 3) addr_sram_act_for_19 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  18*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_19 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 18*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[18] == load_a) & !addr_sram_act_re_end_20) begin
                         addr_sram_act_re_end_19 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_20 <= 0;
                                case (padding_crl_mem[18])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_20 <= addr_sram_act_for_20 + 1;
                                        else addr_sram_act_for_20 <= addr_sram_act_for_20;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_20 <= addr_sram_act_for_20 + 1;
                                        else addr_sram_act_for_20 <= addr_sram_act_for_20;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_20 <= addr_sram_act_for_20 + 1;
                                        else addr_sram_act_for_20 <= addr_sram_act_for_20;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_20 <= addr_sram_act_for_20 + 1;
                                        else addr_sram_act_for_20 <= addr_sram_act_for_20;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_20 <= addr_sram_act_for_20 + 1;
                                        else addr_sram_act_for_20 <= addr_sram_act_for_20;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_20 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[18] == 0) addr_sram_act_for_20 <=  addr_sram_act_for_20 - 3;
                                    else if ((padding_crl_mem[18] == 1)  | (padding_crl_mem[18] == 2) ) addr_sram_act_for_20 <= ADDR_DCNN1_SRAM_ACT_INIT + 19*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[18] == 3) addr_sram_act_for_20 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  19*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_20 <= ADDR_DCNN1_SRAM_ACT_INIT + 19*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[18] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[18] == 0) addr_sram_act_for_20 <=  addr_sram_act_for_20 - 3;
                                        else if ((padding_crl_mem[18] == 1)  | (padding_crl_mem[18] == 2) ) addr_sram_act_for_20 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 19*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[18] == 3) addr_sram_act_for_20 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  19*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_20 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 19*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[18] == 0) addr_sram_act_for_20 <=  addr_sram_act_for_20 - 3;
                                        else if ((padding_crl_mem[18] == 1)  | (padding_crl_mem[18] == 2) ) addr_sram_act_for_20 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 19*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[18] == 3) addr_sram_act_for_20 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  19*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_20 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 19*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[19] == load_a) & !addr_sram_act_re_end_21) begin
                         addr_sram_act_re_end_20 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_21 <= 0;
                                case (padding_crl_mem[19])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_21 <= addr_sram_act_for_21 + 1;
                                        else addr_sram_act_for_21 <= addr_sram_act_for_21;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_21 <= addr_sram_act_for_21 + 1;
                                        else addr_sram_act_for_21 <= addr_sram_act_for_21;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_21 <= addr_sram_act_for_21 + 1;
                                        else addr_sram_act_for_21 <= addr_sram_act_for_21;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_21 <= addr_sram_act_for_21 + 1;
                                        else addr_sram_act_for_21 <= addr_sram_act_for_21;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_21 <= addr_sram_act_for_21 + 1;
                                        else addr_sram_act_for_21 <= addr_sram_act_for_21;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_21 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[19] == 0) addr_sram_act_for_21 <=  addr_sram_act_for_21 - 3;
                                    else if ((padding_crl_mem[19] == 1)  | (padding_crl_mem[19] == 2) ) addr_sram_act_for_21 <= ADDR_DCNN1_SRAM_ACT_INIT + 20*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[19] == 3) addr_sram_act_for_21 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  20*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_21 <= ADDR_DCNN1_SRAM_ACT_INIT + 20*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[19] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[19] == 0) addr_sram_act_for_21 <=  addr_sram_act_for_21 - 3;
                                        else if ((padding_crl_mem[19] == 1)  | (padding_crl_mem[19] == 2) ) addr_sram_act_for_21 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 20*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[19] == 3) addr_sram_act_for_21 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  20*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_21 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 20*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[19] == 0) addr_sram_act_for_21 <=  addr_sram_act_for_21 - 3;
                                        else if ((padding_crl_mem[19] == 1)  | (padding_crl_mem[19] == 2) ) addr_sram_act_for_21 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 20*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[19] == 3) addr_sram_act_for_21 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  20*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_21 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 20*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[20] == load_a) & !addr_sram_act_re_end_22) begin
                         addr_sram_act_re_end_21 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_22 <= 0;
                                case (padding_crl_mem[20])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_22 <= addr_sram_act_for_22 + 1;
                                        else addr_sram_act_for_22 <= addr_sram_act_for_22;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_22 <= addr_sram_act_for_22 + 1;
                                        else addr_sram_act_for_22 <= addr_sram_act_for_22;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_22 <= addr_sram_act_for_22 + 1;
                                        else addr_sram_act_for_22 <= addr_sram_act_for_22;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_22 <= addr_sram_act_for_22 + 1;
                                        else addr_sram_act_for_22 <= addr_sram_act_for_22;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_22 <= addr_sram_act_for_22 + 1;
                                        else addr_sram_act_for_22 <= addr_sram_act_for_22;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_22 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[20] == 0) addr_sram_act_for_22 <=  addr_sram_act_for_22 - 3;
                                    else if ((padding_crl_mem[20] == 1)  | (padding_crl_mem[20] == 2) ) addr_sram_act_for_22 <= ADDR_DCNN1_SRAM_ACT_INIT + 21*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[20] == 3) addr_sram_act_for_22 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  21*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_22 <= ADDR_DCNN1_SRAM_ACT_INIT + 21*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[20] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[20] == 0) addr_sram_act_for_22 <=  addr_sram_act_for_22 - 3;
                                        else if ((padding_crl_mem[20] == 1)  | (padding_crl_mem[20] == 2) ) addr_sram_act_for_22 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 21*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[20] == 3) addr_sram_act_for_22 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  21*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_22 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 21*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[20] == 0) addr_sram_act_for_22 <=  addr_sram_act_for_22 - 3;
                                        else if ((padding_crl_mem[20] == 1)  | (padding_crl_mem[20] == 2) ) addr_sram_act_for_22 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 21*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[20] == 3) addr_sram_act_for_22 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  21*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_22 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 21*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[21] == load_a) & !addr_sram_act_re_end_23) begin
                         addr_sram_act_re_end_22 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_23 <= 0;
                                case (padding_crl_mem[21])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_23 <= addr_sram_act_for_23 + 1;
                                        else addr_sram_act_for_23 <= addr_sram_act_for_23;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_23 <= addr_sram_act_for_23 + 1;
                                        else addr_sram_act_for_23 <= addr_sram_act_for_23;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_23 <= addr_sram_act_for_23 + 1;
                                        else addr_sram_act_for_23 <= addr_sram_act_for_23;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_23 <= addr_sram_act_for_23 + 1;
                                        else addr_sram_act_for_23 <= addr_sram_act_for_23;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_23 <= addr_sram_act_for_23 + 1;
                                        else addr_sram_act_for_23 <= addr_sram_act_for_23;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_23 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[21] == 0) addr_sram_act_for_23 <=  addr_sram_act_for_23 - 3;
                                    else if ((padding_crl_mem[21] == 1)  | (padding_crl_mem[21] == 2) ) addr_sram_act_for_23 <= ADDR_DCNN1_SRAM_ACT_INIT + 22*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[21] == 3) addr_sram_act_for_23 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  22*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_23 <= ADDR_DCNN1_SRAM_ACT_INIT + 22*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[21] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[21] == 0) addr_sram_act_for_23 <=  addr_sram_act_for_23 - 3;
                                        else if ((padding_crl_mem[21] == 1)  | (padding_crl_mem[21] == 2) ) addr_sram_act_for_23 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 22*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[21] == 3) addr_sram_act_for_23 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  22*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_23 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 22*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[21] == 0) addr_sram_act_for_23 <=  addr_sram_act_for_23 - 3;
                                        else if ((padding_crl_mem[21] == 1)  | (padding_crl_mem[21] == 2) ) addr_sram_act_for_23 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 22*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[21] == 3) addr_sram_act_for_23 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  22*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_23 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 22*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[22] == load_a) & !addr_sram_act_re_end_24) begin
                         addr_sram_act_re_end_23 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_24 <= 0;
                                case (padding_crl_mem[22])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_24 <= addr_sram_act_for_24 + 1;
                                        else addr_sram_act_for_24 <= addr_sram_act_for_24;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_24 <= addr_sram_act_for_24 + 1;
                                        else addr_sram_act_for_24 <= addr_sram_act_for_24;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_24 <= addr_sram_act_for_24 + 1;
                                        else addr_sram_act_for_24 <= addr_sram_act_for_24;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_24 <= addr_sram_act_for_24 + 1;
                                        else addr_sram_act_for_24 <= addr_sram_act_for_24;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_24 <= addr_sram_act_for_24 + 1;
                                        else addr_sram_act_for_24 <= addr_sram_act_for_24;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_24 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[22] == 0) addr_sram_act_for_24 <=  addr_sram_act_for_24 - 3;
                                    else if ((padding_crl_mem[22] == 1)  | (padding_crl_mem[22] == 2) ) addr_sram_act_for_24 <= ADDR_DCNN1_SRAM_ACT_INIT + 23*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[22] == 3) addr_sram_act_for_24 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  23*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_24 <= ADDR_DCNN1_SRAM_ACT_INIT + 23*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[22] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[22] == 0) addr_sram_act_for_24 <=  addr_sram_act_for_24 - 3;
                                        else if ((padding_crl_mem[22] == 1)  | (padding_crl_mem[22] == 2) ) addr_sram_act_for_24 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 23*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[22] == 3) addr_sram_act_for_24 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  23*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_24 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 23*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[22] == 0) addr_sram_act_for_24 <=  addr_sram_act_for_24 - 3;
                                        else if ((padding_crl_mem[22] == 1)  | (padding_crl_mem[22] == 2) ) addr_sram_act_for_24 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 23*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[22] == 3) addr_sram_act_for_24 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  23*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_24 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 23*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[23] == load_a) & !addr_sram_act_re_end_25) begin
                         addr_sram_act_re_end_24 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_25 <= 0;
                                case (padding_crl_mem[23])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_25 <= addr_sram_act_for_25 + 1;
                                        else addr_sram_act_for_25 <= addr_sram_act_for_25;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_25 <= addr_sram_act_for_25 + 1;
                                        else addr_sram_act_for_25 <= addr_sram_act_for_25;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_25 <= addr_sram_act_for_25 + 1;
                                        else addr_sram_act_for_25 <= addr_sram_act_for_25;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_25 <= addr_sram_act_for_25 + 1;
                                        else addr_sram_act_for_25 <= addr_sram_act_for_25;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_25 <= addr_sram_act_for_25 + 1;
                                        else addr_sram_act_for_25 <= addr_sram_act_for_25;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_25 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[23] == 0) addr_sram_act_for_25 <=  addr_sram_act_for_25 - 3;
                                    else if ((padding_crl_mem[23] == 1)  | (padding_crl_mem[23] == 2) ) addr_sram_act_for_25 <= ADDR_DCNN1_SRAM_ACT_INIT + 24*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[23] == 3) addr_sram_act_for_25 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  24*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_25 <= ADDR_DCNN1_SRAM_ACT_INIT + 24*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[23] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[23] == 0) addr_sram_act_for_25 <=  addr_sram_act_for_25 - 3;
                                        else if ((padding_crl_mem[23] == 1)  | (padding_crl_mem[23] == 2) ) addr_sram_act_for_25 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 24*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[23] == 3) addr_sram_act_for_25 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  24*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_25 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 24*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[23] == 0) addr_sram_act_for_25 <=  addr_sram_act_for_25 - 3;
                                        else if ((padding_crl_mem[23] == 1)  | (padding_crl_mem[23] == 2) ) addr_sram_act_for_25 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 24*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[23] == 3) addr_sram_act_for_25 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  24*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_25 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 24*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[24] == load_a) & !addr_sram_act_re_end_26) begin
                         addr_sram_act_re_end_25 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_26 <= 0;
                                case (padding_crl_mem[24])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_26 <= addr_sram_act_for_26 + 1;
                                        else addr_sram_act_for_26 <= addr_sram_act_for_26;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_26 <= addr_sram_act_for_26 + 1;
                                        else addr_sram_act_for_26 <= addr_sram_act_for_26;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_26 <= addr_sram_act_for_26 + 1;
                                        else addr_sram_act_for_26 <= addr_sram_act_for_26;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_26 <= addr_sram_act_for_26 + 1;
                                        else addr_sram_act_for_26 <= addr_sram_act_for_26;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_26 <= addr_sram_act_for_26 + 1;
                                        else addr_sram_act_for_26 <= addr_sram_act_for_26;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_26 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[24] == 0) addr_sram_act_for_26 <=  addr_sram_act_for_26 - 3;
                                    else if ((padding_crl_mem[24] == 1)  | (padding_crl_mem[24] == 2) ) addr_sram_act_for_26 <= ADDR_DCNN1_SRAM_ACT_INIT + 25*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[24] == 3) addr_sram_act_for_26 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  25*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_26 <= ADDR_DCNN1_SRAM_ACT_INIT + 25*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[24] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[24] == 0) addr_sram_act_for_26 <=  addr_sram_act_for_26 - 3;
                                        else if ((padding_crl_mem[24] == 1)  | (padding_crl_mem[24] == 2) ) addr_sram_act_for_26 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 25*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[24] == 3) addr_sram_act_for_26 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  25*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_26 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 25*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[24] == 0) addr_sram_act_for_26 <=  addr_sram_act_for_26 - 3;
                                        else if ((padding_crl_mem[24] == 1)  | (padding_crl_mem[24] == 2) ) addr_sram_act_for_26 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 25*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[24] == 3) addr_sram_act_for_26 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  25*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_26 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 25*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[25] == load_a) & !addr_sram_act_re_end_27) begin
                         addr_sram_act_re_end_26 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_27 <= 0;
                                case (padding_crl_mem[25])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_27 <= addr_sram_act_for_27 + 1;
                                        else addr_sram_act_for_27 <= addr_sram_act_for_27;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_27 <= addr_sram_act_for_27 + 1;
                                        else addr_sram_act_for_27 <= addr_sram_act_for_27;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_27 <= addr_sram_act_for_27 + 1;
                                        else addr_sram_act_for_27 <= addr_sram_act_for_27;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_27 <= addr_sram_act_for_27 + 1;
                                        else addr_sram_act_for_27 <= addr_sram_act_for_27;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_27 <= addr_sram_act_for_27 + 1;
                                        else addr_sram_act_for_27 <= addr_sram_act_for_27;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_27 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[25] == 0) addr_sram_act_for_27 <=  addr_sram_act_for_27 - 3;
                                    else if ((padding_crl_mem[25] == 1)  | (padding_crl_mem[25] == 2) ) addr_sram_act_for_27 <= ADDR_DCNN1_SRAM_ACT_INIT + 26*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[25] == 3) addr_sram_act_for_27 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  26*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_27 <= ADDR_DCNN1_SRAM_ACT_INIT + 26*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[25] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[25] == 0) addr_sram_act_for_27 <=  addr_sram_act_for_27 - 3;
                                        else if ((padding_crl_mem[25] == 1)  | (padding_crl_mem[25] == 2) ) addr_sram_act_for_27 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 26*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[25] == 3) addr_sram_act_for_27 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  26*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_27 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 26*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[25] == 0) addr_sram_act_for_27 <=  addr_sram_act_for_27 - 3;
                                        else if ((padding_crl_mem[25] == 1)  | (padding_crl_mem[25] == 2) ) addr_sram_act_for_27 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 26*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[25] == 3) addr_sram_act_for_27 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  26*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_27 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 26*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[26] == load_a) & !addr_sram_act_re_end_28) begin
                         addr_sram_act_re_end_27 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_28 <= 0;
                                case (padding_crl_mem[26])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_28 <= addr_sram_act_for_28 + 1;
                                        else addr_sram_act_for_28 <= addr_sram_act_for_28;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_28 <= addr_sram_act_for_28 + 1;
                                        else addr_sram_act_for_28 <= addr_sram_act_for_28;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_28 <= addr_sram_act_for_28 + 1;
                                        else addr_sram_act_for_28 <= addr_sram_act_for_28;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_28 <= addr_sram_act_for_28 + 1;
                                        else addr_sram_act_for_28 <= addr_sram_act_for_28;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_28 <= addr_sram_act_for_28 + 1;
                                        else addr_sram_act_for_28 <= addr_sram_act_for_28;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_28 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[26] == 0) addr_sram_act_for_28 <=  addr_sram_act_for_28 - 3;
                                    else if ((padding_crl_mem[26] == 1)  | (padding_crl_mem[26] == 2) ) addr_sram_act_for_28 <= ADDR_DCNN1_SRAM_ACT_INIT + 27*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[26] == 3) addr_sram_act_for_28 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  27*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_28 <= ADDR_DCNN1_SRAM_ACT_INIT + 27*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[26] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[26] == 0) addr_sram_act_for_28 <=  addr_sram_act_for_28 - 3;
                                        else if ((padding_crl_mem[26] == 1)  | (padding_crl_mem[26] == 2) ) addr_sram_act_for_28 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 27*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[26] == 3) addr_sram_act_for_28 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  27*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_28 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 27*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[26] == 0) addr_sram_act_for_28 <=  addr_sram_act_for_28 - 3;
                                        else if ((padding_crl_mem[26] == 1)  | (padding_crl_mem[26] == 2) ) addr_sram_act_for_28 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 27*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[26] == 3) addr_sram_act_for_28 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  27*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_28 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 27*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[27] == load_a) & !addr_sram_act_re_end_29) begin
                         addr_sram_act_re_end_28 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_29 <= 0;
                                case (padding_crl_mem[27])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_29 <= addr_sram_act_for_29 + 1;
                                        else addr_sram_act_for_29 <= addr_sram_act_for_29;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_29 <= addr_sram_act_for_29 + 1;
                                        else addr_sram_act_for_29 <= addr_sram_act_for_29;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_29 <= addr_sram_act_for_29 + 1;
                                        else addr_sram_act_for_29 <= addr_sram_act_for_29;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_29 <= addr_sram_act_for_29 + 1;
                                        else addr_sram_act_for_29 <= addr_sram_act_for_29;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_29 <= addr_sram_act_for_29 + 1;
                                        else addr_sram_act_for_29 <= addr_sram_act_for_29;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_29 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[27] == 0) addr_sram_act_for_29 <=  addr_sram_act_for_29 - 3;
                                    else if ((padding_crl_mem[27] == 1)  | (padding_crl_mem[27] == 2) ) addr_sram_act_for_29 <= ADDR_DCNN1_SRAM_ACT_INIT + 28*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[27] == 3) addr_sram_act_for_29 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  28*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_29 <= ADDR_DCNN1_SRAM_ACT_INIT + 28*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[27] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[27] == 0) addr_sram_act_for_29 <=  addr_sram_act_for_29 - 3;
                                        else if ((padding_crl_mem[27] == 1)  | (padding_crl_mem[27] == 2) ) addr_sram_act_for_29 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 28*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[27] == 3) addr_sram_act_for_29 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  28*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_29 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 28*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[27] == 0) addr_sram_act_for_29 <=  addr_sram_act_for_29 - 3;
                                        else if ((padding_crl_mem[27] == 1)  | (padding_crl_mem[27] == 2) ) addr_sram_act_for_29 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 28*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[27] == 3) addr_sram_act_for_29 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  28*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_29 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 28*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[28] == load_a) & !addr_sram_act_re_end_30) begin
                         addr_sram_act_re_end_29 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_30 <= 0;
                                case (padding_crl_mem[28])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_30 <= addr_sram_act_for_30 + 1;
                                        else addr_sram_act_for_30 <= addr_sram_act_for_30;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_30 <= addr_sram_act_for_30 + 1;
                                        else addr_sram_act_for_30 <= addr_sram_act_for_30;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_30 <= addr_sram_act_for_30 + 1;
                                        else addr_sram_act_for_30 <= addr_sram_act_for_30;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_30 <= addr_sram_act_for_30 + 1;
                                        else addr_sram_act_for_30 <= addr_sram_act_for_30;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_30 <= addr_sram_act_for_30 + 1;
                                        else addr_sram_act_for_30 <= addr_sram_act_for_30;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_30 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[28] == 0) addr_sram_act_for_30 <=  addr_sram_act_for_30 - 3;
                                    else if ((padding_crl_mem[28] == 1)  | (padding_crl_mem[28] == 2) ) addr_sram_act_for_30 <= ADDR_DCNN1_SRAM_ACT_INIT + 29*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[28] == 3) addr_sram_act_for_30 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  29*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_30 <= ADDR_DCNN1_SRAM_ACT_INIT + 29*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[28] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[28] == 0) addr_sram_act_for_30 <=  addr_sram_act_for_30 - 3;
                                        else if ((padding_crl_mem[28] == 1)  | (padding_crl_mem[28] == 2) ) addr_sram_act_for_30 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 29*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[28] == 3) addr_sram_act_for_30 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  29*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_30 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 29*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[28] == 0) addr_sram_act_for_30 <=  addr_sram_act_for_30 - 3;
                                        else if ((padding_crl_mem[28] == 1)  | (padding_crl_mem[28] == 2) ) addr_sram_act_for_30 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 29*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[28] == 3) addr_sram_act_for_30 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  29*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_30 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 29*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[29] == load_a) & !addr_sram_act_re_end_31) begin
                         addr_sram_act_re_end_30 <= 0;
                         addr_sram_act_re_end_32 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_31 <= 0;
                                case (padding_crl_mem[29])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_31 <= addr_sram_act_for_31 + 1;
                                        else addr_sram_act_for_31 <= addr_sram_act_for_31;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_31 <= addr_sram_act_for_31 + 1;
                                        else addr_sram_act_for_31 <= addr_sram_act_for_31;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_31 <= addr_sram_act_for_31 + 1;
                                        else addr_sram_act_for_31 <= addr_sram_act_for_31;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_31 <= addr_sram_act_for_31 + 1;
                                        else addr_sram_act_for_31 <= addr_sram_act_for_31;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_31 <= addr_sram_act_for_31 + 1;
                                        else addr_sram_act_for_31 <= addr_sram_act_for_31;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_31 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[29] == 0) addr_sram_act_for_31 <=  addr_sram_act_for_31 - 3;
                                    else if ((padding_crl_mem[29] == 1)  | (padding_crl_mem[29] == 2) ) addr_sram_act_for_31 <= ADDR_DCNN1_SRAM_ACT_INIT + 30*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[29] == 3) addr_sram_act_for_31 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  30*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_31 <= ADDR_DCNN1_SRAM_ACT_INIT + 30*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[29] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[29] == 0) addr_sram_act_for_31 <=  addr_sram_act_for_31 - 3;
                                        else if ((padding_crl_mem[29] == 1)  | (padding_crl_mem[29] == 2) ) addr_sram_act_for_31 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 30*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[29] == 3) addr_sram_act_for_31 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  30*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_31 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 30*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[29] == 0) addr_sram_act_for_31 <=  addr_sram_act_for_31 - 3;
                                        else if ((padding_crl_mem[29] == 1)  | (padding_crl_mem[29] == 2) ) addr_sram_act_for_31 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 30*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[29] == 3) addr_sram_act_for_31 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  30*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_31 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 30*CNN12_LENGTH_IN ;
                                    end
                                end
                            end   
                        end
                     else if ((conv_state_mem[30] == load_a) & !addr_sram_act_re_end_32) begin
                         addr_sram_act_re_end_31 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_32 <= 0;
                                case (padding_crl_mem[30])
                                    0: begin
                                        if (cnt_re_sram < CNN_KS - 1) addr_sram_act_for_32 <= addr_sram_act_for_32 + 1;
                                        else addr_sram_act_for_32 <= addr_sram_act_for_32;
                                    end
                                    1: begin
                                        if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_32 <= addr_sram_act_for_32 + 1;
                                        else addr_sram_act_for_32 <= addr_sram_act_for_32;
                                    end
                                    2:begin
                                        if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS - 1)) addr_sram_act_for_32 <= addr_sram_act_for_32 + 1;
                                        else addr_sram_act_for_32 <= addr_sram_act_for_32;                           
                                    end
                                    3:begin
                                        if  (cnt_re_sram < CNN_KS-2) addr_sram_act_for_32 <= addr_sram_act_for_32 + 1;
                                        else addr_sram_act_for_32 <= addr_sram_act_for_32;                                  
                                    end
                                    4:begin
                                        if  (cnt_re_sram < CNN_KS-3) addr_sram_act_for_32 <= addr_sram_act_for_32 + 1;
                                        else addr_sram_act_for_32 <= addr_sram_act_for_32;                             
                                    end
                                endcase
                            end  
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_32 <= 1;
                                if (decoder_top_state == cnn11) begin
                                    if (padding_crl_mem[30] == 0) addr_sram_act_for_32 <=  addr_sram_act_for_32 - 3;
                                    else if ((padding_crl_mem[30] == 1)  | (padding_crl_mem[30] == 2) ) addr_sram_act_for_32 <= ADDR_DCNN1_SRAM_ACT_INIT + 31*CNN11_LENGTH_IN;
                                    else if (padding_crl_mem[30] == 3) addr_sram_act_for_32 <= ADDR_DCNN1_SRAM_ACT_INIT + 125+  31*CNN11_LENGTH_IN;
                                    else  addr_sram_act_for_32 <= ADDR_DCNN1_SRAM_ACT_INIT + 31*CNN11_LENGTH_IN ; 
                                end
                                else if (decoder_top_state == cnn12) begin
                                    if (cnt_cho_mem[30] < CNN11_CHOUT/2 ) begin
                                        if (padding_crl_mem[30] == 0) addr_sram_act_for_32 <=  addr_sram_act_for_32 - 3;
                                        else if ((padding_crl_mem[30] == 1)  | (padding_crl_mem[30] == 2) ) addr_sram_act_for_32 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 31*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[30] == 3) addr_sram_act_for_32 <= ADDR_CNN11_SRAM_ACT_INIT_0 +  125+  31*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_32 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 31*CNN12_LENGTH_IN ;  
                                    end
                                    else begin
                                        if (padding_crl_mem[30] == 0) addr_sram_act_for_32 <=  addr_sram_act_for_32 - 3;
                                        else if ((padding_crl_mem[30] == 1)  | (padding_crl_mem[30] == 2) ) addr_sram_act_for_32 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 31*CNN12_LENGTH_IN;
                                        else if (padding_crl_mem[30] == 3) addr_sram_act_for_32 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 125+  31*CNN12_LENGTH_IN;
                                        else addr_sram_act_for_32 <= ADDR_CNN11_SRAM_ACT_INIT_1 + 31*CNN12_LENGTH_IN ;
                                    end
                                end
                            end 

                        end
                    else if (layer_done) begin
                        if (decoder_top_state == cnn11) begin
                            addr_sram_act_for_1 <= ADDR_CNN11_SRAM_ACT_INIT_0;
                            addr_sram_act_for_2 <= ADDR_CNN11_SRAM_ACT_INIT_0 + CNN11_LENGTH_IN;
                            addr_sram_act_for_3 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 2 * CNN11_LENGTH_IN;
                            addr_sram_act_for_4 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 3 * CNN11_LENGTH_IN;
                            addr_sram_act_for_5 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 4 * CNN11_LENGTH_IN;
                            addr_sram_act_for_6 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 5 * CNN11_LENGTH_IN;
                            addr_sram_act_for_7 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 6 * CNN11_LENGTH_IN;
                            addr_sram_act_for_8 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 7 * CNN11_LENGTH_IN;
                            addr_sram_act_for_9 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 8 * CNN11_LENGTH_IN;
                            addr_sram_act_for_10 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 9 * CNN11_LENGTH_IN;
                            addr_sram_act_for_11 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 10 * CNN11_LENGTH_IN;
                            addr_sram_act_for_12 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 11 * CNN11_LENGTH_IN;
                            addr_sram_act_for_13 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 12 * CNN11_LENGTH_IN;
                            addr_sram_act_for_14 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 13 * CNN11_LENGTH_IN;
                            addr_sram_act_for_15 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 14 * CNN11_LENGTH_IN;
                            addr_sram_act_for_16 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 15 * CNN11_LENGTH_IN;
                            addr_sram_act_for_17 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 16 * CNN11_LENGTH_IN;
                            addr_sram_act_for_18 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 17 * CNN11_LENGTH_IN;
                            addr_sram_act_for_19 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 18 * CNN11_LENGTH_IN;
                            addr_sram_act_for_20 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 19 * CNN11_LENGTH_IN;
                            addr_sram_act_for_21 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 20 * CNN11_LENGTH_IN;
                            addr_sram_act_for_22 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 21 * CNN11_LENGTH_IN;
                            addr_sram_act_for_23 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 22 * CNN11_LENGTH_IN;
                            addr_sram_act_for_24 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 23 * CNN11_LENGTH_IN;
                            addr_sram_act_for_25 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 24 * CNN11_LENGTH_IN;
                            addr_sram_act_for_26 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 25 * CNN11_LENGTH_IN;
                            addr_sram_act_for_27 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 26 * CNN11_LENGTH_IN;
                            addr_sram_act_for_28 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 27 * CNN11_LENGTH_IN;
                            addr_sram_act_for_29 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 28 * CNN11_LENGTH_IN;
                            addr_sram_act_for_30 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 29 * CNN11_LENGTH_IN;
                            addr_sram_act_for_31 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 30 * CNN11_LENGTH_IN;
                            addr_sram_act_for_32 <= ADDR_CNN11_SRAM_ACT_INIT_0 + 31 * CNN11_LENGTH_IN;                       
                        end
                        else if (decoder_top_state == cnn12) begin
                            addr_sram_act_for_1 <= ADDR_CNN12_SRAM_ACT_INIT;
                            addr_sram_act_for_2 <= ADDR_CNN12_SRAM_ACT_INIT + CNN11_LENGTH_IN;
                            addr_sram_act_for_3 <= ADDR_CNN12_SRAM_ACT_INIT + 2 * CNN11_LENGTH_IN;
                            addr_sram_act_for_4 <= ADDR_CNN12_SRAM_ACT_INIT + 3 * CNN11_LENGTH_IN;
                            addr_sram_act_for_5 <= ADDR_CNN12_SRAM_ACT_INIT + 4 * CNN11_LENGTH_IN;
                            addr_sram_act_for_6 <= ADDR_CNN12_SRAM_ACT_INIT + 5 * CNN11_LENGTH_IN;
                            addr_sram_act_for_7 <= ADDR_CNN12_SRAM_ACT_INIT + 6 * CNN11_LENGTH_IN;
                            addr_sram_act_for_8 <= ADDR_CNN12_SRAM_ACT_INIT + 7 * CNN11_LENGTH_IN;
                            addr_sram_act_for_9 <= ADDR_CNN12_SRAM_ACT_INIT + 8 * CNN11_LENGTH_IN;
                            addr_sram_act_for_10 <= ADDR_CNN12_SRAM_ACT_INIT + 9 * CNN11_LENGTH_IN;
                            addr_sram_act_for_11 <= ADDR_CNN12_SRAM_ACT_INIT + 10 * CNN11_LENGTH_IN;
                            addr_sram_act_for_12 <= ADDR_CNN12_SRAM_ACT_INIT + 11 * CNN11_LENGTH_IN;
                            addr_sram_act_for_13 <= ADDR_CNN12_SRAM_ACT_INIT + 12 * CNN11_LENGTH_IN;
                            addr_sram_act_for_14 <= ADDR_CNN12_SRAM_ACT_INIT + 13 * CNN11_LENGTH_IN;
                            addr_sram_act_for_15 <= ADDR_CNN12_SRAM_ACT_INIT + 14 * CNN11_LENGTH_IN;
                            addr_sram_act_for_16 <= ADDR_CNN12_SRAM_ACT_INIT + 15 * CNN11_LENGTH_IN;
                            addr_sram_act_for_17 <= ADDR_CNN12_SRAM_ACT_INIT + 0 * CNN11_LENGTH_IN;
                            addr_sram_act_for_18 <= ADDR_CNN12_SRAM_ACT_INIT + 1 * CNN11_LENGTH_IN;
                            addr_sram_act_for_19 <= ADDR_CNN12_SRAM_ACT_INIT + 2 * CNN11_LENGTH_IN;
                            addr_sram_act_for_20 <= ADDR_CNN12_SRAM_ACT_INIT + 3 * CNN11_LENGTH_IN;
                            addr_sram_act_for_21 <= ADDR_CNN12_SRAM_ACT_INIT + 4 * CNN11_LENGTH_IN;
                            addr_sram_act_for_22 <= ADDR_CNN12_SRAM_ACT_INIT + 5 * CNN11_LENGTH_IN;
                            addr_sram_act_for_23 <= ADDR_CNN12_SRAM_ACT_INIT + 6 * CNN11_LENGTH_IN;
                            addr_sram_act_for_24 <= ADDR_CNN12_SRAM_ACT_INIT + 7 * CNN11_LENGTH_IN;
                            addr_sram_act_for_25 <= ADDR_CNN12_SRAM_ACT_INIT + 8 * CNN11_LENGTH_IN;
                            addr_sram_act_for_26 <= ADDR_CNN12_SRAM_ACT_INIT + 9 * CNN11_LENGTH_IN;
                            addr_sram_act_for_27 <= ADDR_CNN12_SRAM_ACT_INIT + 10 * CNN11_LENGTH_IN;
                            addr_sram_act_for_28 <= ADDR_CNN12_SRAM_ACT_INIT + 11 * CNN11_LENGTH_IN;
                            addr_sram_act_for_29 <= ADDR_CNN12_SRAM_ACT_INIT + 12 * CNN11_LENGTH_IN;
                            addr_sram_act_for_30 <= ADDR_CNN12_SRAM_ACT_INIT + 13 * CNN11_LENGTH_IN;
                            addr_sram_act_for_31 <= ADDR_CNN12_SRAM_ACT_INIT + 14 * CNN11_LENGTH_IN;
                            addr_sram_act_for_32 <= ADDR_CNN12_SRAM_ACT_INIT + 15 * CNN11_LENGTH_IN;                         
                        end    
                        else if (decoder_top_state == cnn21)  begin
                            addr_sram_act_for_1 <= ADDR_CNN21_SRAM_ACT_INIT;
                            addr_sram_act_for_2 <= ADDR_CNN21_SRAM_ACT_INIT + CNN22_LENGTH_IN;
                            addr_sram_act_for_3 <= ADDR_CNN21_SRAM_ACT_INIT + 2 * CNN22_LENGTH_IN;
                            addr_sram_act_for_4 <= ADDR_CNN21_SRAM_ACT_INIT + 3 * CNN22_LENGTH_IN;
                            addr_sram_act_for_5 <= ADDR_CNN21_SRAM_ACT_INIT + 4 * CNN22_LENGTH_IN;
                            addr_sram_act_for_6 <= ADDR_CNN21_SRAM_ACT_INIT + 5 * CNN22_LENGTH_IN;
                            addr_sram_act_for_7 <= ADDR_CNN21_SRAM_ACT_INIT + 6 * CNN22_LENGTH_IN;
                            addr_sram_act_for_8 <= ADDR_CNN21_SRAM_ACT_INIT + 7 * CNN22_LENGTH_IN;
                            addr_sram_act_for_9 <= ADDR_CNN21_SRAM_ACT_INIT + 0 * CNN22_LENGTH_IN;
                            addr_sram_act_for_10 <= ADDR_CNN21_SRAM_ACT_INIT + 1 * CNN22_LENGTH_IN;
                            addr_sram_act_for_11 <= ADDR_CNN21_SRAM_ACT_INIT + 2 * CNN22_LENGTH_IN;
                            addr_sram_act_for_12 <= ADDR_CNN21_SRAM_ACT_INIT + 3 * CNN22_LENGTH_IN;
                            addr_sram_act_for_13 <= ADDR_CNN21_SRAM_ACT_INIT + 4 * CNN22_LENGTH_IN;
                            addr_sram_act_for_14 <= ADDR_CNN21_SRAM_ACT_INIT + 5 * CNN22_LENGTH_IN;
                            addr_sram_act_for_15 <= ADDR_CNN21_SRAM_ACT_INIT + 6 * CNN22_LENGTH_IN;
                            addr_sram_act_for_16 <= ADDR_CNN21_SRAM_ACT_INIT + 7 * CNN22_LENGTH_IN;
                            addr_sram_act_for_17 <= ADDR_CNN21_SRAM_ACT_INIT + 0 * CNN22_LENGTH_IN;
                            addr_sram_act_for_18 <= ADDR_CNN21_SRAM_ACT_INIT + 1 * CNN22_LENGTH_IN;
                            addr_sram_act_for_19 <= ADDR_CNN21_SRAM_ACT_INIT + 2 * CNN22_LENGTH_IN;
                            addr_sram_act_for_20 <= ADDR_CNN21_SRAM_ACT_INIT + 3 * CNN22_LENGTH_IN;
                            addr_sram_act_for_21 <= ADDR_CNN21_SRAM_ACT_INIT + 4 * CNN22_LENGTH_IN;
                            addr_sram_act_for_22 <= ADDR_CNN21_SRAM_ACT_INIT + 5 * CNN22_LENGTH_IN;
                            addr_sram_act_for_23 <= ADDR_CNN21_SRAM_ACT_INIT + 6 * CNN22_LENGTH_IN;
                            addr_sram_act_for_24 <= ADDR_CNN21_SRAM_ACT_INIT + 7 * CNN22_LENGTH_IN;
                            addr_sram_act_for_25 <= ADDR_CNN21_SRAM_ACT_INIT + 0 * CNN22_LENGTH_IN;
                            addr_sram_act_for_26 <= ADDR_CNN21_SRAM_ACT_INIT + 1 * CNN22_LENGTH_IN;
                            addr_sram_act_for_27 <= ADDR_CNN21_SRAM_ACT_INIT + 2 * CNN22_LENGTH_IN;
                            addr_sram_act_for_28 <= ADDR_CNN21_SRAM_ACT_INIT + 3 * CNN22_LENGTH_IN;
                            addr_sram_act_for_29 <= ADDR_CNN21_SRAM_ACT_INIT + 4 * CNN22_LENGTH_IN;
                            addr_sram_act_for_30 <= ADDR_CNN21_SRAM_ACT_INIT + 5 * CNN22_LENGTH_IN;
                            addr_sram_act_for_31 <= ADDR_CNN21_SRAM_ACT_INIT + 6 * CNN22_LENGTH_IN;
                            addr_sram_act_for_32 <= ADDR_CNN21_SRAM_ACT_INIT + 7 * CNN22_LENGTH_IN;                         
                        end
                        else if (decoder_top_state == cnn22)  begin
                            addr_sram_act_for_1 <= ADDR_LSTM10_SRAM_ACT_INIT;
                            addr_sram_act_for_2 <= ADDR_LSTM10_SRAM_ACT_INIT + 1;
                            addr_sram_act_for_3 <= ADDR_LSTM10_SRAM_ACT_INIT + 2;
                            addr_sram_act_for_4 <= ADDR_LSTM10_SRAM_ACT_INIT + 3;
                            addr_sram_act_for_5 <= ADDR_LSTM10_SRAM_ACT_INIT + 4;
                            addr_sram_act_for_6 <= ADDR_LSTM10_SRAM_ACT_INIT + 5;
                            addr_sram_act_for_7 <= ADDR_LSTM10_SRAM_ACT_INIT + 6;
                            addr_sram_act_for_8 <= ADDR_LSTM10_SRAM_ACT_INIT + 7;
                            addr_sram_act_for_9 <= ADDR_LSTM10_SRAM_ACT_INIT + 8;
                            addr_sram_act_for_10 <= ADDR_LSTM10_SRAM_ACT_INIT + 9;
                            addr_sram_act_for_11 <= ADDR_LSTM10_SRAM_ACT_INIT + 10;
                            addr_sram_act_for_12 <= ADDR_LSTM10_SRAM_ACT_INIT + 11;
                            addr_sram_act_for_13 <= ADDR_LSTM10_SRAM_ACT_INIT + 12;
                            addr_sram_act_for_14 <= ADDR_LSTM10_SRAM_ACT_INIT + 13;
                            addr_sram_act_for_15 <= ADDR_LSTM10_SRAM_ACT_INIT + 14;
                            addr_sram_act_for_16 <= ADDR_LSTM10_SRAM_ACT_INIT + 15;
                            addr_sram_act_for_17 <= ADDR_LSTM10_SRAM_ACT_INIT + 16;
                            addr_sram_act_for_18 <= ADDR_LSTM10_SRAM_ACT_INIT + 17;
                            addr_sram_act_for_19 <= ADDR_LSTM10_SRAM_ACT_INIT + 18;
                            addr_sram_act_for_20 <= ADDR_LSTM10_SRAM_ACT_INIT + 19;
                            addr_sram_act_for_21 <= ADDR_LSTM10_SRAM_ACT_INIT + 20;
                            addr_sram_act_for_22 <= ADDR_LSTM10_SRAM_ACT_INIT + 21;
                            addr_sram_act_for_23 <= ADDR_LSTM10_SRAM_ACT_INIT + 22;
                            addr_sram_act_for_24 <= ADDR_LSTM10_SRAM_ACT_INIT + 23;
                            addr_sram_act_for_25 <= ADDR_LSTM10_SRAM_ACT_INIT + 24;
                            addr_sram_act_for_26 <= ADDR_LSTM10_SRAM_ACT_INIT + 25;
                            addr_sram_act_for_27 <= ADDR_LSTM10_SRAM_ACT_INIT + 26;
                            addr_sram_act_for_28 <= ADDR_LSTM10_SRAM_ACT_INIT + 27;
                            addr_sram_act_for_29 <= ADDR_LSTM10_SRAM_ACT_INIT + 28;
                            addr_sram_act_for_30 <= ADDR_LSTM10_SRAM_ACT_INIT + 29;
                            addr_sram_act_for_31 <= ADDR_LSTM10_SRAM_ACT_INIT + 30;
                            addr_sram_act_for_32 <= ADDR_LSTM10_SRAM_ACT_INIT + 31;                        
                        end
                        else ;
                    end
                else ;
                
                end
            end
            else if (decoder_top_state == dcnn2) begin
                if (sram_act_we) begin
                    cnt_re_sram  <= cnt_re_sram;  
                end
                else begin
                    if ((conv_state == load_a) & !addr_sram_act_re_end_1) begin
                        if (cnt_re_sram != SPAD_DEPTH-1) begin
                            cnt_re_sram <= cnt_re_sram + 1;
                            addr_sram_act_re_end_1 <= 0;
                            case (padding_crl)
                                0: begin
                                    if (~is_odd) begin
                                        if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))  begin
                                            addr_sram_act_for_1 <=  addr_sram_act_for_1 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_1 <= addr_sram_act_for_1;
                                        end
                                    end
                                    else begin
                                        if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) begin
                                            addr_sram_act_for_1 <=  addr_sram_act_for_1 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_1 <= addr_sram_act_for_1; 
                                        end                                   
                                    end
                                end
                                1: begin
                                    if (cnt_re_sram == 5) begin
                                        addr_sram_act_for_1 <= addr_sram_act_for_1 + 1;
                                    end
                                    else begin
                                        addr_sram_act_for_1 <= addr_sram_act_for_1;
                                    end
                                end
                                2:begin
                                    if ((cnt_re_sram == 4)| (cnt_re_sram == 6)) begin
                                        addr_sram_act_for_1 <= addr_sram_act_for_1 + 1;
                                    end  
                                    else begin
                                        addr_sram_act_for_1 <= addr_sram_act_for_1;
                                    end                              
                                end
                                3:begin
                                    if ((cnt_re_sram == 3)| (cnt_re_sram == 5)) begin
                                        addr_sram_act_for_1 <=   addr_sram_act_for_1 + 1;
                                    end   
                                    else begin
                                        addr_sram_act_for_1 <= addr_sram_act_for_1;
                                    end                                  
                                end
                                4:begin
                                    if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                        addr_sram_act_for_1 <=  addr_sram_act_for_1 + 1;
                                    end 
                                    else begin
                                        addr_sram_act_for_1 <= addr_sram_act_for_1;
                                    end                                
                                end
                                5:begin
                                    if ((cnt_re_sram == 1)|(cnt_re_sram == 3)) begin
                                        addr_sram_act_for_1 <= addr_sram_act_for_1 + 1;
                                    end
                                    else begin
                                        addr_sram_act_for_1 <= addr_sram_act_for_1;
                                    end                             
                                end
                                6: begin
                                    if ((cnt_re_sram == 0)|(cnt_re_sram == 2)) begin
                                        addr_sram_act_for_1 <=  addr_sram_act_for_1 + 1;
                                    end 
                                    else begin
                                        addr_sram_act_for_1 <= addr_sram_act_for_1;
                                    end                                   
                                end
                            endcase
                        end
                        else begin
                            cnt_re_sram <= 0;
                            addr_sram_act_re_end_1 <= 1;
                            if (padding_crl == 0) begin
                                addr_sram_act_for_1 <= (is_odd)?  addr_sram_act_for_1 - 3:  addr_sram_act_for_1 - 3 + 1;
                            end
                            else if ((padding_crl == 1)  | (padding_crl == 2) | (padding_crl == 3)) begin
                                addr_sram_act_for_1 <= ADDR_CNN12_SRAM_ACT_INIT + 0*DCNN2_LENGTH_IN ;
                            end
                            else if ((padding_crl == 4)  | (padding_crl == 5) ) begin
                                addr_sram_act_for_1 <= ADDR_CNN12_SRAM_ACT_INIT + 125 + 0*DCNN2_LENGTH_IN  ;
                            end
                            else begin
                                addr_sram_act_for_1 <= ADDR_CNN12_SRAM_ACT_INIT  + 0*DCNN2_LENGTH_IN  ;
                            end

                        end
                    end  
                    else if ((conv_state_mem[0] == load_a) & !addr_sram_act_re_end_2) begin
                            addr_sram_act_re_end_1 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_2 <= 0;
                                case (padding_crl_mem[0])
                                    0: begin
                                        if (~is_odd_mem[0]) begin
                                            if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))  begin
                                                addr_sram_act_for_2 <=  addr_sram_act_for_2 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_2 <= addr_sram_act_for_2;
                                            end
                                        end
                                        else begin
                                            if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) begin
                                                addr_sram_act_for_2 <=  addr_sram_act_for_2 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_2 <= addr_sram_act_for_2; 
                                            end                                   
                                        end
                                    end
                                    1: begin
                                        if (cnt_re_sram == 5) begin
                                            addr_sram_act_for_2 <= addr_sram_act_for_2 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_2 <= addr_sram_act_for_2;
                                        end
                                    end
                                    2:begin
                                        if ((cnt_re_sram == 4)| (cnt_re_sram == 6)) begin
                                            addr_sram_act_for_2 <= addr_sram_act_for_2 + 1;
                                        end  
                                        else begin
                                            addr_sram_act_for_2 <= addr_sram_act_for_2;
                                        end                              
                                    end
                                    3:begin
                                        if ((cnt_re_sram == 3)| (cnt_re_sram == 5)) begin
                                            addr_sram_act_for_2 <=   addr_sram_act_for_2 + 1;
                                        end   
                                        else begin
                                            addr_sram_act_for_2 <= addr_sram_act_for_2;
                                        end                                  
                                    end
                                    4:begin
                                        if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                            addr_sram_act_for_2 <=  addr_sram_act_for_2 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_2 <= addr_sram_act_for_2;
                                        end                                
                                    end
                                    5:begin
                                        if ((cnt_re_sram == 1)|(cnt_re_sram == 3)) begin
                                            addr_sram_act_for_2 <= addr_sram_act_for_2 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_2 <= addr_sram_act_for_2;
                                        end                             
                                    end
                                    6: begin
                                        if ((cnt_re_sram == 0)|(cnt_re_sram == 2)) begin
                                            addr_sram_act_for_2 <=  addr_sram_act_for_2 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_2 <= addr_sram_act_for_2;
                                        end                                   
                                    end
                                endcase
                            end
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_2 <= 1;
                                if (padding_crl_mem[0] == 0) begin
                                    addr_sram_act_for_2 <= (is_odd_mem[0])?  addr_sram_act_for_2 - 3:  addr_sram_act_for_2 - 3 + 1;
                                end
                                else if ((padding_crl_mem[0] == 1)  | (padding_crl_mem[0] == 2) | (padding_crl_mem[0] == 3)) begin
                                    addr_sram_act_for_2 <= ADDR_CNN12_SRAM_ACT_INIT + 1*DCNN2_LENGTH_IN ;
                                end
                                else if ((padding_crl_mem[0] == 4)  | (padding_crl_mem[0] == 5) ) begin
                                    addr_sram_act_for_2 <= ADDR_CNN12_SRAM_ACT_INIT + 125 + 1*DCNN2_LENGTH_IN  ;
                                end
                                else begin
                                    addr_sram_act_for_2 <= ADDR_CNN12_SRAM_ACT_INIT  + 1*DCNN2_LENGTH_IN  ;
                                end
                            end
                        end
                    else if ((conv_state_mem[1] == load_a) & !addr_sram_act_re_end_3) begin
                        addr_sram_act_re_end_2 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_3 <= 0;
                                case (padding_crl_mem[1])
                                    0: begin
                                        if (~is_odd_mem[1]) begin
                                            if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))  begin
                                                addr_sram_act_for_3 <=  addr_sram_act_for_3 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_3 <= addr_sram_act_for_3;
                                            end
                                        end
                                        else begin
                                            if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) begin
                                                addr_sram_act_for_3 <=  addr_sram_act_for_3 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_3 <= addr_sram_act_for_3; 
                                            end                                   
                                        end
                                    end
                                    1: begin
                                        if (cnt_re_sram == 5) begin
                                            addr_sram_act_for_3 <= addr_sram_act_for_3 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_3 <= addr_sram_act_for_3;
                                        end
                                    end
                                    2:begin
                                        if ((cnt_re_sram == 4)| (cnt_re_sram == 6)) begin
                                            addr_sram_act_for_3 <= addr_sram_act_for_3 + 1;
                                        end  
                                        else begin
                                            addr_sram_act_for_3 <= addr_sram_act_for_3;
                                        end                              
                                    end
                                    3:begin
                                        if ((cnt_re_sram == 3)| (cnt_re_sram == 5)) begin
                                            addr_sram_act_for_3 <=   addr_sram_act_for_3 + 1;
                                        end   
                                        else begin
                                            addr_sram_act_for_3 <= addr_sram_act_for_3;
                                        end                                  
                                    end
                                    4:begin
                                        if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                            addr_sram_act_for_3 <=  addr_sram_act_for_3 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_3 <= addr_sram_act_for_3;
                                        end                                
                                    end
                                    5:begin
                                        if ((cnt_re_sram == 1)|(cnt_re_sram == 3)) begin
                                            addr_sram_act_for_3 <= addr_sram_act_for_3 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_3 <= addr_sram_act_for_3;
                                        end                             
                                    end
                                    6: begin
                                        if ((cnt_re_sram == 0)|(cnt_re_sram == 2)) begin
                                            addr_sram_act_for_3 <=  addr_sram_act_for_3 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_3 <= addr_sram_act_for_3;
                                        end                                   
                                    end
                                endcase
                            end
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_3 <= 1;
                                if (padding_crl_mem[1] == 0) begin
                                    addr_sram_act_for_3 <= (is_odd_mem[1])?  addr_sram_act_for_3 - 3:  addr_sram_act_for_3 - 3 + 1;
                                end
                                else if ((padding_crl_mem[1] == 1)  | (padding_crl_mem[1] == 2) | (padding_crl_mem[1] == 3)) begin
                                    addr_sram_act_for_3 <= ADDR_CNN12_SRAM_ACT_INIT + 2*DCNN2_LENGTH_IN ;
                                end
                                else if ((padding_crl_mem[1] == 4)  | (padding_crl_mem[1] == 5) ) begin
                                    addr_sram_act_for_3 <= ADDR_CNN12_SRAM_ACT_INIT + 125 + 2*DCNN2_LENGTH_IN  ;
                                end
                                else begin
                                    addr_sram_act_for_3 <= ADDR_CNN12_SRAM_ACT_INIT  + 2*DCNN2_LENGTH_IN  ;
                                end
                            end
                        end
                    else if ((conv_state_mem[2] == load_a) & !addr_sram_act_re_end_4) begin
                        addr_sram_act_re_end_3 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_4 <= 0;
                                case (padding_crl_mem[2])
                                    0: begin
                                        if (~is_odd_mem[2]) begin
                                            if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))  begin
                                                addr_sram_act_for_4 <=  addr_sram_act_for_4 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_4 <= addr_sram_act_for_4;
                                            end
                                        end
                                        else begin
                                            if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) begin
                                                addr_sram_act_for_4 <=  addr_sram_act_for_4 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_4 <= addr_sram_act_for_4; 
                                            end                                   
                                        end
                                    end
                                    1: begin
                                        if (cnt_re_sram == 5) begin
                                            addr_sram_act_for_4 <= addr_sram_act_for_4 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_4 <= addr_sram_act_for_4;
                                        end
                                    end
                                    2:begin
                                        if ((cnt_re_sram == 4)| (cnt_re_sram == 6)) begin
                                            addr_sram_act_for_4 <= addr_sram_act_for_4 + 1;
                                        end  
                                        else begin
                                            addr_sram_act_for_4 <= addr_sram_act_for_4;
                                        end                              
                                    end
                                    3:begin
                                        if ((cnt_re_sram == 3)| (cnt_re_sram == 5)) begin
                                            addr_sram_act_for_4 <=   addr_sram_act_for_4 + 1;
                                        end   
                                        else begin
                                            addr_sram_act_for_4 <= addr_sram_act_for_4;
                                        end                                  
                                    end
                                    4:begin
                                        if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                            addr_sram_act_for_4 <=  addr_sram_act_for_4 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_4 <= addr_sram_act_for_4;
                                        end                                
                                    end
                                    5:begin
                                        if ((cnt_re_sram == 1)|(cnt_re_sram == 3)) begin
                                            addr_sram_act_for_4 <= addr_sram_act_for_4 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_4 <= addr_sram_act_for_4;
                                        end                             
                                    end
                                    6: begin
                                        if ((cnt_re_sram == 0)|(cnt_re_sram == 2)) begin
                                            addr_sram_act_for_4 <=  addr_sram_act_for_4 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_4 <= addr_sram_act_for_4;
                                        end                                   
                                    end
                                endcase
                            end
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_4 <= 1;
                                if (padding_crl_mem[2] == 0) begin
                                    addr_sram_act_for_4 <= (is_odd_mem[2])?  addr_sram_act_for_4 - 3:  addr_sram_act_for_4 - 3 + 1;
                                end
                                else if ((padding_crl_mem[2] == 1)  | (padding_crl_mem[2] == 2) | (padding_crl_mem[2] == 3)) begin
                                    addr_sram_act_for_4 <= ADDR_CNN12_SRAM_ACT_INIT + 3*DCNN2_LENGTH_IN ;
                                end
                                else if ((padding_crl_mem[2] == 4)  | (padding_crl_mem[2] == 5) ) begin
                                    addr_sram_act_for_4 <= ADDR_CNN12_SRAM_ACT_INIT + 125 + 3*DCNN2_LENGTH_IN  ;
                                end
                                else begin
                                    addr_sram_act_for_4 <= ADDR_CNN12_SRAM_ACT_INIT  + 3*DCNN2_LENGTH_IN  ;
                                end
                            end
                        end
                    else if ((conv_state_mem[3] == load_a) & !addr_sram_act_re_end_5) begin
                        addr_sram_act_re_end_4 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_5 <= 0;
                                case (padding_crl_mem[3])
                                    0: begin
                                        if (~is_odd_mem[3]) begin
                                            if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))  begin
                                                addr_sram_act_for_5 <=  addr_sram_act_for_5 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_5 <= addr_sram_act_for_5;
                                            end
                                        end
                                        else begin
                                            if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) begin
                                                addr_sram_act_for_5 <=  addr_sram_act_for_5 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_5 <= addr_sram_act_for_5; 
                                            end                                   
                                        end
                                    end
                                    1: begin
                                        if (cnt_re_sram == 5) begin
                                            addr_sram_act_for_5 <= addr_sram_act_for_5 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_5 <= addr_sram_act_for_5;
                                        end
                                    end
                                    2:begin
                                        if ((cnt_re_sram == 4)| (cnt_re_sram == 6)) begin
                                            addr_sram_act_for_5 <= addr_sram_act_for_5 + 1;
                                        end  
                                        else begin
                                            addr_sram_act_for_5 <= addr_sram_act_for_5;
                                        end                              
                                    end
                                    3:begin
                                        if ((cnt_re_sram == 3)| (cnt_re_sram == 5)) begin
                                            addr_sram_act_for_5 <=   addr_sram_act_for_5 + 1;
                                        end   
                                        else begin
                                            addr_sram_act_for_5 <= addr_sram_act_for_5;
                                        end                                  
                                    end
                                    4:begin
                                        if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                            addr_sram_act_for_5 <=  addr_sram_act_for_5 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_5 <= addr_sram_act_for_5;
                                        end                                
                                    end
                                    5:begin
                                        if ((cnt_re_sram == 1)|(cnt_re_sram == 3)) begin
                                            addr_sram_act_for_5 <= addr_sram_act_for_5 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_5 <= addr_sram_act_for_5;
                                        end                             
                                    end
                                    6: begin
                                        if ((cnt_re_sram == 0)|(cnt_re_sram == 2)) begin
                                            addr_sram_act_for_5 <=  addr_sram_act_for_5 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_5 <= addr_sram_act_for_5;
                                        end                                   
                                    end
                                endcase
                            end
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_5 <= 1;
                                if (padding_crl_mem[3] == 0) begin
                                    addr_sram_act_for_5 <= (is_odd_mem[3])?  addr_sram_act_for_5 - 3:  addr_sram_act_for_5 - 3 + 1;
                                end
                                else if ((padding_crl_mem[3] == 1)  | (padding_crl_mem[3] == 2) | (padding_crl_mem[3] == 3)) begin
                                    addr_sram_act_for_5 <= ADDR_CNN12_SRAM_ACT_INIT + 4*DCNN2_LENGTH_IN ;
                                end
                                else if ((padding_crl_mem[3] == 4)  | (padding_crl_mem[3] == 5) ) begin
                                    addr_sram_act_for_5 <= ADDR_CNN12_SRAM_ACT_INIT + 125 + 4*DCNN2_LENGTH_IN  ;
                                end
                                else begin
                                    addr_sram_act_for_5 <= ADDR_CNN12_SRAM_ACT_INIT  + 4*DCNN2_LENGTH_IN  ;
                                end
                            end
                        end
                    else if ((conv_state_mem[4] == load_a) & !addr_sram_act_re_end_6) begin
                            addr_sram_act_re_end_5 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_6 <= 0;
                                case (padding_crl_mem[4])
                                    0: begin
                                        if (~is_odd_mem[4]) begin
                                            if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))  begin
                                                addr_sram_act_for_6 <=  addr_sram_act_for_6 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_6 <= addr_sram_act_for_6;
                                            end
                                        end
                                        else begin
                                            if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) begin
                                                addr_sram_act_for_6 <=  addr_sram_act_for_6 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_6 <= addr_sram_act_for_6; 
                                            end                                   
                                        end
                                    end
                                    1: begin
                                        if (cnt_re_sram == 5) begin
                                            addr_sram_act_for_6 <= addr_sram_act_for_6 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_6 <= addr_sram_act_for_6;
                                        end
                                    end
                                    2:begin
                                        if ((cnt_re_sram == 4)| (cnt_re_sram == 6)) begin
                                            addr_sram_act_for_6 <= addr_sram_act_for_6 + 1;
                                        end  
                                        else begin
                                            addr_sram_act_for_6 <= addr_sram_act_for_6;
                                        end                              
                                    end
                                    3:begin
                                        if ((cnt_re_sram == 3)| (cnt_re_sram == 5)) begin
                                            addr_sram_act_for_6 <=   addr_sram_act_for_6 + 1;
                                        end   
                                        else begin
                                            addr_sram_act_for_6 <= addr_sram_act_for_6;
                                        end                                  
                                    end
                                    4:begin
                                        if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                            addr_sram_act_for_6 <=  addr_sram_act_for_6 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_6 <= addr_sram_act_for_6;
                                        end                                
                                    end
                                    5:begin
                                        if ((cnt_re_sram == 1)|(cnt_re_sram == 3)) begin
                                            addr_sram_act_for_6 <= addr_sram_act_for_6 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_6 <= addr_sram_act_for_6;
                                        end                             
                                    end
                                    6: begin
                                        if ((cnt_re_sram == 0)|(cnt_re_sram == 2)) begin
                                            addr_sram_act_for_6 <=  addr_sram_act_for_6 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_6 <= addr_sram_act_for_6;
                                        end                                   
                                    end
                                endcase
                            end
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_6 <= 1;
                                if (padding_crl_mem[4] == 0) begin
                                    addr_sram_act_for_6 <= (is_odd_mem[4])?  addr_sram_act_for_6 - 3:  addr_sram_act_for_6 - 3 + 1;
                                end
                                else if ((padding_crl_mem[4] == 1)  | (padding_crl_mem[4] == 2) | (padding_crl_mem[4] == 3)) begin
                                    addr_sram_act_for_6 <= ADDR_CNN12_SRAM_ACT_INIT + 5*DCNN2_LENGTH_IN ;
                                end
                                else if ((padding_crl_mem[4] == 4)  | (padding_crl_mem[4] == 5) ) begin
                                    addr_sram_act_for_6 <= ADDR_CNN12_SRAM_ACT_INIT + 125 + 5*DCNN2_LENGTH_IN  ;
                                end
                                else begin
                                    addr_sram_act_for_6 <= ADDR_CNN12_SRAM_ACT_INIT  + 5*DCNN2_LENGTH_IN  ;
                                end
                            end
                        end
                    else if ((conv_state_mem[5] == load_a) & !addr_sram_act_re_end_7) begin
                            addr_sram_act_re_end_6 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_7 <= 0;
                                case (padding_crl_mem[5])
                                    0: begin
                                        if (~is_odd_mem[5]) begin
                                            if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))  begin
                                                addr_sram_act_for_7 <=  addr_sram_act_for_7 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_7 <= addr_sram_act_for_7;
                                            end
                                        end
                                        else begin
                                            if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) begin
                                                addr_sram_act_for_7 <=  addr_sram_act_for_7 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_7 <= addr_sram_act_for_7; 
                                            end                                   
                                        end
                                    end
                                    1: begin
                                        if (cnt_re_sram == 5) begin
                                            addr_sram_act_for_7 <= addr_sram_act_for_7 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_7 <= addr_sram_act_for_7;
                                        end
                                    end
                                    2:begin
                                        if ((cnt_re_sram == 4)| (cnt_re_sram == 6)) begin
                                            addr_sram_act_for_7 <= addr_sram_act_for_7 + 1;
                                        end  
                                        else begin
                                            addr_sram_act_for_7 <= addr_sram_act_for_7;
                                        end                              
                                    end
                                    3:begin
                                        if ((cnt_re_sram == 3)| (cnt_re_sram == 5)) begin
                                            addr_sram_act_for_7 <=   addr_sram_act_for_7 + 1;
                                        end   
                                        else begin
                                            addr_sram_act_for_7 <= addr_sram_act_for_7;
                                        end                                  
                                    end
                                    4:begin
                                        if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                            addr_sram_act_for_7 <=  addr_sram_act_for_7 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_7 <= addr_sram_act_for_7;
                                        end                                
                                    end
                                    5:begin
                                        if ((cnt_re_sram == 1)|(cnt_re_sram == 3)) begin
                                            addr_sram_act_for_7 <= addr_sram_act_for_7 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_7 <= addr_sram_act_for_7;
                                        end                             
                                    end
                                    6: begin
                                        if ((cnt_re_sram == 0)|(cnt_re_sram == 2)) begin
                                            addr_sram_act_for_7 <=  addr_sram_act_for_7 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_7 <= addr_sram_act_for_7;
                                        end                                   
                                    end
                                endcase
                            end
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_7 <= 1;
                                if (padding_crl_mem[5] == 0) begin
                                    addr_sram_act_for_7 <= (is_odd_mem[5])?  addr_sram_act_for_7 - 3:  addr_sram_act_for_7 - 3 + 1;
                                end
                                else if ((padding_crl_mem[5] == 1)  | (padding_crl_mem[5] == 2) | (padding_crl_mem[5] == 3)) begin
                                    addr_sram_act_for_7 <= ADDR_CNN12_SRAM_ACT_INIT + 6*DCNN2_LENGTH_IN ;
                                end
                                else if ((padding_crl_mem[5] == 4)  | (padding_crl_mem[5] == 5) ) begin
                                    addr_sram_act_for_7 <= ADDR_CNN12_SRAM_ACT_INIT + 125 + 6*DCNN2_LENGTH_IN  ;
                                end
                                else begin
                                    addr_sram_act_for_7 <= ADDR_CNN12_SRAM_ACT_INIT  + 6*DCNN2_LENGTH_IN  ;
                                end
                            end
                        end
                    else if ((conv_state_mem[6] == load_a) & !addr_sram_act_re_end_8) begin
                            addr_sram_act_re_end_7 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_8 <= 0;
                                case (padding_crl_mem[6])
                                    0: begin
                                        if (~is_odd_mem[6]) begin
                                            if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))  begin
                                                addr_sram_act_for_8 <=  addr_sram_act_for_8 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_8 <= addr_sram_act_for_8;
                                            end
                                        end
                                        else begin
                                            if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) begin
                                                addr_sram_act_for_8 <=  addr_sram_act_for_8 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_8 <= addr_sram_act_for_8; 
                                            end                                   
                                        end
                                    end
                                    1: begin
                                        if (cnt_re_sram == 5) begin
                                            addr_sram_act_for_8 <= addr_sram_act_for_8 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_8 <= addr_sram_act_for_8;
                                        end
                                    end
                                    2:begin
                                        if ((cnt_re_sram == 4)| (cnt_re_sram == 6)) begin
                                            addr_sram_act_for_8 <= addr_sram_act_for_8 + 1;
                                        end  
                                        else begin
                                            addr_sram_act_for_8 <= addr_sram_act_for_8;
                                        end                              
                                    end
                                    3:begin
                                        if ((cnt_re_sram == 3)| (cnt_re_sram == 5)) begin
                                            addr_sram_act_for_8 <=   addr_sram_act_for_8 + 1;
                                        end   
                                        else begin
                                            addr_sram_act_for_8 <= addr_sram_act_for_8;
                                        end                                  
                                    end
                                    4:begin
                                        if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                            addr_sram_act_for_8 <=  addr_sram_act_for_8 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_8 <= addr_sram_act_for_8;
                                        end                                
                                    end
                                    5:begin
                                        if ((cnt_re_sram == 1)|(cnt_re_sram == 3)) begin
                                            addr_sram_act_for_8 <= addr_sram_act_for_8 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_8 <= addr_sram_act_for_8;
                                        end                             
                                    end
                                    6: begin
                                        if ((cnt_re_sram == 0)|(cnt_re_sram == 2)) begin
                                            addr_sram_act_for_8 <=  addr_sram_act_for_8 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_8 <= addr_sram_act_for_8;
                                        end                                   
                                    end
                                endcase
                            end
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_8 <= 1;
                                if (padding_crl_mem[6] == 0) begin
                                    addr_sram_act_for_8 <= (is_odd_mem[6])?  addr_sram_act_for_8 - 3:  addr_sram_act_for_8 - 3 + 1;
                                end
                                else if ((padding_crl_mem[6] == 1)  | (padding_crl_mem[6] == 2) | (padding_crl_mem[6] == 3)) begin
                                    addr_sram_act_for_8 <= ADDR_CNN12_SRAM_ACT_INIT + 7*DCNN2_LENGTH_IN ;
                                end
                                else if ((padding_crl_mem[6] == 4)  | (padding_crl_mem[6] == 5) ) begin
                                    addr_sram_act_for_8 <= ADDR_CNN12_SRAM_ACT_INIT + 125 + 7*DCNN2_LENGTH_IN  ;
                                end
                                else begin
                                    addr_sram_act_for_8 <= ADDR_CNN12_SRAM_ACT_INIT  + 7*DCNN2_LENGTH_IN  ;
                                end
                            end
                        end
                    else if ((conv_state_mem[7] == load_a) & !addr_sram_act_re_end_9) begin
                            addr_sram_act_re_end_8 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_9 <= 0;
                                case (padding_crl_mem[7])
                                    0: begin
                                        if (~is_odd_mem[7]) begin
                                            if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))  begin
                                                addr_sram_act_for_9 <=  addr_sram_act_for_9 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_9 <= addr_sram_act_for_9;
                                            end
                                        end
                                        else begin
                                            if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) begin
                                                addr_sram_act_for_9 <=  addr_sram_act_for_9 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_9 <= addr_sram_act_for_9; 
                                            end                                   
                                        end
                                    end
                                    1: begin
                                        if (cnt_re_sram == 5) begin
                                            addr_sram_act_for_9 <= addr_sram_act_for_9 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_9 <= addr_sram_act_for_9;
                                        end
                                    end
                                    2:begin
                                        if ((cnt_re_sram == 4)| (cnt_re_sram == 6)) begin
                                            addr_sram_act_for_9 <= addr_sram_act_for_9 + 1;
                                        end  
                                        else begin
                                            addr_sram_act_for_9 <= addr_sram_act_for_9;
                                        end                              
                                    end
                                    3:begin
                                        if ((cnt_re_sram == 3)| (cnt_re_sram == 5)) begin
                                            addr_sram_act_for_9 <=   addr_sram_act_for_9 + 1;
                                        end   
                                        else begin
                                            addr_sram_act_for_9 <= addr_sram_act_for_9;
                                        end                                  
                                    end
                                    4:begin
                                        if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                            addr_sram_act_for_9 <=  addr_sram_act_for_9 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_9 <= addr_sram_act_for_9;
                                        end                                
                                    end
                                    5:begin
                                        if ((cnt_re_sram == 1)|(cnt_re_sram == 3)) begin
                                            addr_sram_act_for_9 <= addr_sram_act_for_9 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_9 <= addr_sram_act_for_9;
                                        end                             
                                    end
                                    6: begin
                                        if ((cnt_re_sram == 0)|(cnt_re_sram == 2)) begin
                                            addr_sram_act_for_9 <=  addr_sram_act_for_9 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_9 <= addr_sram_act_for_9;
                                        end                                   
                                    end
                                endcase
                            end
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_9 <= 1;
                                if (padding_crl_mem[7] == 0) begin
                                    addr_sram_act_for_9 <= (is_odd_mem[7])?  addr_sram_act_for_9 - 3:  addr_sram_act_for_9 - 3 + 1;
                                end
                                else if ((padding_crl_mem[7] == 1)  | (padding_crl_mem[7] == 2) | (padding_crl_mem[7] == 3)) begin
                                    addr_sram_act_for_9 <= ADDR_CNN12_SRAM_ACT_INIT + 8*DCNN2_LENGTH_IN ;
                                end
                                else if ((padding_crl_mem[7] == 4)  | (padding_crl_mem[7] == 5) ) begin
                                    addr_sram_act_for_9 <= ADDR_CNN12_SRAM_ACT_INIT + 125 + 8*DCNN2_LENGTH_IN  ;
                                end
                                else begin
                                    addr_sram_act_for_9 <= ADDR_CNN12_SRAM_ACT_INIT  + 8*DCNN2_LENGTH_IN  ;
                                end
                            end
                        end
                    else if ((conv_state_mem[8] == load_a) & !addr_sram_act_re_end_10) begin
                        addr_sram_act_re_end_9 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_10 <= 0;
                                case (padding_crl_mem[8])
                                    0: begin
                                        if (~is_odd_mem[8]) begin
                                            if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))  begin
                                                addr_sram_act_for_10 <=  addr_sram_act_for_10 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_10 <= addr_sram_act_for_10;
                                            end
                                        end
                                        else begin
                                            if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) begin
                                                addr_sram_act_for_10 <=  addr_sram_act_for_10 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_10 <= addr_sram_act_for_10; 
                                            end                                   
                                        end
                                    end
                                    1: begin
                                        if (cnt_re_sram == 5) begin
                                            addr_sram_act_for_10 <= addr_sram_act_for_10 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_10 <= addr_sram_act_for_10;
                                        end
                                    end
                                    2:begin
                                        if ((cnt_re_sram == 4)| (cnt_re_sram == 6)) begin
                                            addr_sram_act_for_10 <= addr_sram_act_for_10 + 1;
                                        end  
                                        else begin
                                            addr_sram_act_for_10 <= addr_sram_act_for_10;
                                        end                              
                                    end
                                    3:begin
                                        if ((cnt_re_sram == 3)| (cnt_re_sram == 5)) begin
                                            addr_sram_act_for_10 <=   addr_sram_act_for_10 + 1;
                                        end   
                                        else begin
                                            addr_sram_act_for_10 <= addr_sram_act_for_10;
                                        end                                  
                                    end
                                    4:begin
                                        if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                            addr_sram_act_for_10 <=  addr_sram_act_for_10 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_10 <= addr_sram_act_for_10;
                                        end                                
                                    end
                                    5:begin
                                        if ((cnt_re_sram == 1)|(cnt_re_sram == 3)) begin
                                            addr_sram_act_for_10 <= addr_sram_act_for_10 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_10 <= addr_sram_act_for_10;
                                        end                             
                                    end
                                    6: begin
                                        if ((cnt_re_sram == 0)|(cnt_re_sram == 2)) begin
                                            addr_sram_act_for_10 <=  addr_sram_act_for_10 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_10 <= addr_sram_act_for_10;
                                        end                                   
                                    end
                                endcase
                            end
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_10 <= 1;
                                if (padding_crl_mem[8] == 0) begin
                                    addr_sram_act_for_10 <= (is_odd_mem[8])?  addr_sram_act_for_10 - 3:  addr_sram_act_for_10 - 3 + 1;
                                end
                                else if ((padding_crl_mem[8] == 1)  | (padding_crl_mem[8] == 2) | (padding_crl_mem[8] == 3)) begin
                                    addr_sram_act_for_10 <= ADDR_CNN12_SRAM_ACT_INIT + 9*DCNN2_LENGTH_IN ;
                                end
                                else if ((padding_crl_mem[8] == 4)  | (padding_crl_mem[8] == 5) ) begin
                                    addr_sram_act_for_10 <= ADDR_CNN12_SRAM_ACT_INIT + 125 + 9*DCNN2_LENGTH_IN  ;
                                end
                                else begin
                                    addr_sram_act_for_10 <= ADDR_CNN12_SRAM_ACT_INIT  + 9*DCNN2_LENGTH_IN  ;
                                end
                            end
                        end
                    else if ((conv_state_mem[9] == load_a) & !addr_sram_act_re_end_11) begin
                        addr_sram_act_re_end_10 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_11 <= 0;
                                case (padding_crl_mem[9])
                                    0: begin
                                        if (~is_odd_mem[9]) begin
                                            if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))  begin
                                                addr_sram_act_for_11 <=  addr_sram_act_for_11 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_11 <= addr_sram_act_for_11;
                                            end
                                        end
                                        else begin
                                            if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) begin
                                                addr_sram_act_for_11 <=  addr_sram_act_for_11 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_11 <= addr_sram_act_for_11; 
                                            end                                   
                                        end
                                    end
                                    1: begin
                                        if (cnt_re_sram == 5) begin
                                            addr_sram_act_for_11 <= addr_sram_act_for_11 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_11 <= addr_sram_act_for_11;
                                        end
                                    end
                                    2:begin
                                        if ((cnt_re_sram == 4)| (cnt_re_sram == 6)) begin
                                            addr_sram_act_for_11 <= addr_sram_act_for_11 + 1;
                                        end  
                                        else begin
                                            addr_sram_act_for_11 <= addr_sram_act_for_11;
                                        end                              
                                    end
                                    3:begin
                                        if ((cnt_re_sram == 3)| (cnt_re_sram == 5)) begin
                                            addr_sram_act_for_11 <=   addr_sram_act_for_11 + 1;
                                        end   
                                        else begin
                                            addr_sram_act_for_11 <= addr_sram_act_for_11;
                                        end                                  
                                    end
                                    4:begin
                                        if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                            addr_sram_act_for_11 <=  addr_sram_act_for_11 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_11 <= addr_sram_act_for_11;
                                        end                                
                                    end
                                    5:begin
                                        if ((cnt_re_sram == 1)|(cnt_re_sram == 3)) begin
                                            addr_sram_act_for_11 <= addr_sram_act_for_11 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_11 <= addr_sram_act_for_11;
                                        end                             
                                    end
                                    6: begin
                                        if ((cnt_re_sram == 0)|(cnt_re_sram == 2)) begin
                                            addr_sram_act_for_11 <=  addr_sram_act_for_11 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_11 <= addr_sram_act_for_11;
                                        end                                   
                                    end
                                endcase
                            end
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_11 <= 1;
                                if (padding_crl_mem[9] == 0) begin
                                    addr_sram_act_for_11 <= (is_odd_mem[9])?  addr_sram_act_for_11 - 3:  addr_sram_act_for_11 - 3 + 1;
                                end
                                else if ((padding_crl_mem[9] == 1)  | (padding_crl_mem[9] == 2) | (padding_crl_mem[9] == 3)) begin
                                    addr_sram_act_for_11 <= ADDR_CNN12_SRAM_ACT_INIT + 10*DCNN2_LENGTH_IN ;
                                end
                                else if ((padding_crl_mem[9] == 4)  | (padding_crl_mem[9] == 5) ) begin
                                    addr_sram_act_for_11 <= ADDR_CNN12_SRAM_ACT_INIT + 125 + 10*DCNN2_LENGTH_IN  ;
                                end
                                else begin
                                    addr_sram_act_for_11 <= ADDR_CNN12_SRAM_ACT_INIT  + 10*DCNN2_LENGTH_IN  ;
                                end
                            end
                        end
                    else if ((conv_state_mem[10] == load_a) & !addr_sram_act_re_end_12) begin
                        addr_sram_act_re_end_11 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_12 <= 0;
                                case (padding_crl_mem[10])
                                    0: begin
                                        if (~is_odd_mem[10]) begin
                                            if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))  begin
                                                addr_sram_act_for_12 <=  addr_sram_act_for_12 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_12 <= addr_sram_act_for_12;
                                            end
                                        end
                                        else begin
                                            if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) begin
                                                addr_sram_act_for_12 <=  addr_sram_act_for_12 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_12 <= addr_sram_act_for_12; 
                                            end                                   
                                        end
                                    end
                                    1: begin
                                        if (cnt_re_sram == 5) begin
                                            addr_sram_act_for_12 <= addr_sram_act_for_12 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_12 <= addr_sram_act_for_12;
                                        end
                                    end
                                    2:begin
                                        if ((cnt_re_sram == 4)| (cnt_re_sram == 6)) begin
                                            addr_sram_act_for_12 <= addr_sram_act_for_12 + 1;
                                        end  
                                        else begin
                                            addr_sram_act_for_12 <= addr_sram_act_for_12;
                                        end                              
                                    end
                                    3:begin
                                        if ((cnt_re_sram == 3)| (cnt_re_sram == 5)) begin
                                            addr_sram_act_for_12 <=   addr_sram_act_for_12 + 1;
                                        end   
                                        else begin
                                            addr_sram_act_for_12 <= addr_sram_act_for_12;
                                        end                                  
                                    end
                                    4:begin
                                        if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                            addr_sram_act_for_12 <=  addr_sram_act_for_12 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_12 <= addr_sram_act_for_12;
                                        end                                
                                    end
                                    5:begin
                                        if ((cnt_re_sram == 1)|(cnt_re_sram == 3)) begin
                                            addr_sram_act_for_12 <= addr_sram_act_for_12 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_12 <= addr_sram_act_for_12;
                                        end                             
                                    end
                                    6: begin
                                        if ((cnt_re_sram == 0)|(cnt_re_sram == 2)) begin
                                            addr_sram_act_for_12 <=  addr_sram_act_for_12 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_12 <= addr_sram_act_for_12;
                                        end                                   
                                    end
                                endcase
                            end
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_12 <= 1;
                                if (padding_crl_mem[10] == 0) begin
                                    addr_sram_act_for_12 <= (is_odd_mem[10])?  addr_sram_act_for_12 - 3:  addr_sram_act_for_12 - 3 + 1;
                                end
                                else if ((padding_crl_mem[10] == 1)  | (padding_crl_mem[10] == 2) | (padding_crl_mem[10] == 3)) begin
                                    addr_sram_act_for_12 <= ADDR_CNN12_SRAM_ACT_INIT + 11*DCNN2_LENGTH_IN ;
                                end
                                else if ((padding_crl_mem[10] == 4)  | (padding_crl_mem[10] == 5) ) begin
                                    addr_sram_act_for_12 <= ADDR_CNN12_SRAM_ACT_INIT + 125 + 11*DCNN2_LENGTH_IN  ;
                                end
                                else begin
                                    addr_sram_act_for_12 <= ADDR_CNN12_SRAM_ACT_INIT  + 11*DCNN2_LENGTH_IN  ;
                                end
                            end
                        end
                    else if ((conv_state_mem[11] == load_a) & !addr_sram_act_re_end_13) begin
                        addr_sram_act_re_end_12 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_13 <= 0;
                                case (padding_crl_mem[11])
                                    0: begin
                                        if (~is_odd_mem[11]) begin
                                            if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))  begin
                                                addr_sram_act_for_13 <=  addr_sram_act_for_13 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_13 <= addr_sram_act_for_13;
                                            end
                                        end
                                        else begin
                                            if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) begin
                                                addr_sram_act_for_13 <=  addr_sram_act_for_13 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_13 <= addr_sram_act_for_13; 
                                            end                                   
                                        end
                                    end
                                    1: begin
                                        if (cnt_re_sram == 5) begin
                                            addr_sram_act_for_13 <= addr_sram_act_for_13 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_13 <= addr_sram_act_for_13;
                                        end
                                    end
                                    2:begin
                                        if ((cnt_re_sram == 4)| (cnt_re_sram == 6)) begin
                                            addr_sram_act_for_13 <= addr_sram_act_for_13 + 1;
                                        end  
                                        else begin
                                            addr_sram_act_for_13 <= addr_sram_act_for_13;
                                        end                              
                                    end
                                    3:begin
                                        if ((cnt_re_sram == 3)| (cnt_re_sram == 5)) begin
                                            addr_sram_act_for_13 <=   addr_sram_act_for_13 + 1;
                                        end   
                                        else begin
                                            addr_sram_act_for_13 <= addr_sram_act_for_13;
                                        end                                  
                                    end
                                    4:begin
                                        if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                            addr_sram_act_for_13 <=  addr_sram_act_for_13 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_13 <= addr_sram_act_for_13;
                                        end                                
                                    end
                                    5:begin
                                        if ((cnt_re_sram == 1)|(cnt_re_sram == 3)) begin
                                            addr_sram_act_for_13 <= addr_sram_act_for_13 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_13 <= addr_sram_act_for_13;
                                        end                             
                                    end
                                    6: begin
                                        if ((cnt_re_sram == 0)|(cnt_re_sram == 2)) begin
                                            addr_sram_act_for_13 <=  addr_sram_act_for_13 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_13 <= addr_sram_act_for_13;
                                        end                                   
                                    end
                                endcase
                            end
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_13 <= 1;
                                if (padding_crl_mem[11] == 0) begin
                                    addr_sram_act_for_13 <= (is_odd_mem[11])?  addr_sram_act_for_13 - 3:  addr_sram_act_for_13 - 3 + 1;
                                end
                                else if ((padding_crl_mem[11] == 1)  | (padding_crl_mem[11] == 2) | (padding_crl_mem[11] == 3)) begin
                                    addr_sram_act_for_13 <= ADDR_CNN12_SRAM_ACT_INIT + 12*DCNN2_LENGTH_IN ;
                                end
                                else if ((padding_crl_mem[11] == 4)  | (padding_crl_mem[11] == 5) ) begin
                                    addr_sram_act_for_13 <= ADDR_CNN12_SRAM_ACT_INIT + 125 + 12*DCNN2_LENGTH_IN  ;
                                end
                                else begin
                                    addr_sram_act_for_13 <= ADDR_CNN12_SRAM_ACT_INIT  + 12*DCNN2_LENGTH_IN  ;
                                end
                            end
                        end
                    else if ((conv_state_mem[12] == load_a) & !addr_sram_act_re_end_14) begin
                        addr_sram_act_re_end_13 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_14 <= 0;
                                case (padding_crl_mem[12])
                                    0: begin
                                        if (~is_odd_mem[12]) begin
                                            if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))  begin
                                                addr_sram_act_for_14 <=  addr_sram_act_for_14 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_14 <= addr_sram_act_for_14;
                                            end
                                        end
                                        else begin
                                            if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) begin
                                                addr_sram_act_for_14 <=  addr_sram_act_for_14 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_14 <= addr_sram_act_for_14; 
                                            end                                   
                                        end
                                    end
                                    1: begin
                                        if (cnt_re_sram == 5) begin
                                            addr_sram_act_for_14 <= addr_sram_act_for_14 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_14 <= addr_sram_act_for_14;
                                        end
                                    end
                                    2:begin
                                        if ((cnt_re_sram == 4)| (cnt_re_sram == 6)) begin
                                            addr_sram_act_for_14 <= addr_sram_act_for_14 + 1;
                                        end  
                                        else begin
                                            addr_sram_act_for_14 <= addr_sram_act_for_14;
                                        end                              
                                    end
                                    3:begin
                                        if ((cnt_re_sram == 3)| (cnt_re_sram == 5)) begin
                                            addr_sram_act_for_14 <=   addr_sram_act_for_14 + 1;
                                        end   
                                        else begin
                                            addr_sram_act_for_14 <= addr_sram_act_for_14;
                                        end                                  
                                    end
                                    4:begin
                                        if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                            addr_sram_act_for_14 <=  addr_sram_act_for_14 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_14 <= addr_sram_act_for_14;
                                        end                                
                                    end
                                    5:begin
                                        if ((cnt_re_sram == 1)|(cnt_re_sram == 3)) begin
                                            addr_sram_act_for_14 <= addr_sram_act_for_14 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_14 <= addr_sram_act_for_14;
                                        end                             
                                    end
                                    6: begin
                                        if ((cnt_re_sram == 0)|(cnt_re_sram == 2)) begin
                                            addr_sram_act_for_14 <=  addr_sram_act_for_14 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_14 <= addr_sram_act_for_14;
                                        end                                   
                                    end
                                endcase
                            end
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_14 <= 1;
                                if (padding_crl_mem[12] == 0) begin
                                    addr_sram_act_for_14 <= (is_odd_mem[12])?  addr_sram_act_for_14 - 3:  addr_sram_act_for_14 - 3 + 1;
                                end
                                else if ((padding_crl_mem[12] == 1)  | (padding_crl_mem[12] == 2) | (padding_crl_mem[12] == 3)) begin
                                    addr_sram_act_for_14 <= ADDR_CNN12_SRAM_ACT_INIT + 13*DCNN2_LENGTH_IN ;
                                end
                                else if ((padding_crl_mem[12] == 4)  | (padding_crl_mem[12] == 5) ) begin
                                    addr_sram_act_for_14 <= ADDR_CNN12_SRAM_ACT_INIT + 125 + 13*DCNN2_LENGTH_IN  ;
                                end
                                else begin
                                    addr_sram_act_for_14 <= ADDR_CNN12_SRAM_ACT_INIT  + 13*DCNN2_LENGTH_IN  ;
                                end
                            end
                        end
                    else if ((conv_state_mem[13] == load_a) & !addr_sram_act_re_end_15) begin
                        addr_sram_act_re_end_14 <= 0;
                        addr_sram_act_re_end_16 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_15 <= 0;
                                case (padding_crl_mem[13])
                                    0: begin
                                        if (~is_odd_mem[13]) begin
                                            if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))  begin
                                                addr_sram_act_for_15 <=  addr_sram_act_for_15 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_15 <= addr_sram_act_for_15;
                                            end
                                        end
                                        else begin
                                            if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) begin
                                                addr_sram_act_for_15 <=  addr_sram_act_for_15 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_15 <= addr_sram_act_for_15; 
                                            end                                   
                                        end
                                    end
                                    1: begin
                                        if (cnt_re_sram == 5) begin
                                            addr_sram_act_for_15 <= addr_sram_act_for_15 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_15 <= addr_sram_act_for_15;
                                        end
                                    end
                                    2:begin
                                        if ((cnt_re_sram == 4)| (cnt_re_sram == 6)) begin
                                            addr_sram_act_for_15 <= addr_sram_act_for_15 + 1;
                                        end  
                                        else begin
                                            addr_sram_act_for_15 <= addr_sram_act_for_15;
                                        end                              
                                    end
                                    3:begin
                                        if ((cnt_re_sram == 3)| (cnt_re_sram == 5)) begin
                                            addr_sram_act_for_15 <=   addr_sram_act_for_15 + 1;
                                        end   
                                        else begin
                                            addr_sram_act_for_15 <= addr_sram_act_for_15;
                                        end                                  
                                    end
                                    4:begin
                                        if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                            addr_sram_act_for_15 <=  addr_sram_act_for_15 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_15 <= addr_sram_act_for_15;
                                        end                                
                                    end
                                    5:begin
                                        if ((cnt_re_sram == 1)|(cnt_re_sram == 3)) begin
                                            addr_sram_act_for_15 <= addr_sram_act_for_15 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_15 <= addr_sram_act_for_15;
                                        end                             
                                    end
                                    6: begin
                                        if ((cnt_re_sram == 0)|(cnt_re_sram == 2)) begin
                                            addr_sram_act_for_15 <=  addr_sram_act_for_15 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_15 <= addr_sram_act_for_15;
                                        end                                   
                                    end
                                endcase
                            end
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_15 <= 1;
                                if (padding_crl_mem[13] == 0) begin
                                    addr_sram_act_for_15 <= (is_odd_mem[13])?  addr_sram_act_for_15 - 3:  addr_sram_act_for_15 - 3 + 1;
                                end
                                else if ((padding_crl_mem[13] == 1)  | (padding_crl_mem[13] == 2) | (padding_crl_mem[13] == 3)) begin
                                    addr_sram_act_for_15 <= ADDR_CNN12_SRAM_ACT_INIT + 14*DCNN2_LENGTH_IN ;
                                end
                                else if ((padding_crl_mem[13] == 4)  | (padding_crl_mem[13] == 5) ) begin
                                    addr_sram_act_for_15 <= ADDR_CNN12_SRAM_ACT_INIT + 125 + 14*DCNN2_LENGTH_IN  ;
                                end
                                else begin
                                    addr_sram_act_for_15 <= ADDR_CNN12_SRAM_ACT_INIT  + 14*DCNN2_LENGTH_IN  ;
                                end
                            end
                        end
                    else if ((conv_state_mem[14] == load_a) & !addr_sram_act_re_end_16) begin
                        addr_sram_act_re_end_15 <= 0;
                            if (cnt_re_sram != SPAD_DEPTH-1) begin
                                cnt_re_sram <= cnt_re_sram + 1;
                                addr_sram_act_re_end_16 <= 0;
                                case (padding_crl_mem[14])
                                    0: begin
                                        if (~is_odd_mem[14]) begin
                                            if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5))  begin
                                                addr_sram_act_for_16 <=  addr_sram_act_for_16 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_16 <= addr_sram_act_for_16;
                                            end
                                        end
                                        else begin
                                            if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) begin
                                                addr_sram_act_for_16 <=  addr_sram_act_for_16 + 1;
                                            end
                                            else begin
                                                addr_sram_act_for_16 <= addr_sram_act_for_16; 
                                            end                                   
                                        end
                                    end
                                    1: begin
                                        if (cnt_re_sram == 5) begin
                                            addr_sram_act_for_16 <= addr_sram_act_for_16 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_16 <= addr_sram_act_for_16;
                                        end
                                    end
                                    2:begin
                                        if ((cnt_re_sram == 4)| (cnt_re_sram == 6)) begin
                                            addr_sram_act_for_16 <= addr_sram_act_for_16 + 1;
                                        end  
                                        else begin
                                            addr_sram_act_for_16 <= addr_sram_act_for_16;
                                        end                              
                                    end
                                    3:begin
                                        if ((cnt_re_sram == 3)| (cnt_re_sram == 5)) begin
                                            addr_sram_act_for_16 <=   addr_sram_act_for_16 + 1;
                                        end   
                                        else begin
                                            addr_sram_act_for_16 <= addr_sram_act_for_16;
                                        end                                  
                                    end
                                    4:begin
                                        if ((cnt_re_sram == 2)|(cnt_re_sram == 4)) begin
                                            addr_sram_act_for_16 <=  addr_sram_act_for_16 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_16 <= addr_sram_act_for_16;
                                        end                                
                                    end
                                    5:begin
                                        if ((cnt_re_sram == 1)|(cnt_re_sram == 3)) begin
                                            addr_sram_act_for_16 <= addr_sram_act_for_16 + 1;
                                        end
                                        else begin
                                            addr_sram_act_for_16 <= addr_sram_act_for_16;
                                        end                             
                                    end
                                    6: begin
                                        if ((cnt_re_sram == 0)|(cnt_re_sram == 2)) begin
                                            addr_sram_act_for_16 <=  addr_sram_act_for_16 + 1;
                                        end 
                                        else begin
                                            addr_sram_act_for_16 <= addr_sram_act_for_16;
                                        end                                   
                                    end
                                endcase
                            end
                            else begin
                                cnt_re_sram <= 0;
                                addr_sram_act_re_end_16 <= 1;
                                if (padding_crl_mem[14] == 0) begin
                                    addr_sram_act_for_16 <= (is_odd_mem[14])?  addr_sram_act_for_16 - 3:  addr_sram_act_for_16 - 3 + 1;
                                end
                                else if ((padding_crl_mem[14] == 1)  | (padding_crl_mem[14] == 2) | (padding_crl_mem[14] == 3)) begin
                                    addr_sram_act_for_16 <= ADDR_CNN12_SRAM_ACT_INIT + 15*DCNN2_LENGTH_IN ;
                                end
                                else if ((padding_crl_mem[14] == 4)  | (padding_crl_mem[14] == 5) ) begin
                                    addr_sram_act_for_16 <= ADDR_CNN12_SRAM_ACT_INIT + 125 + 15*DCNN2_LENGTH_IN  ;
                                end
                                else begin
                                    addr_sram_act_for_16 <= ADDR_CNN12_SRAM_ACT_INIT  + 15*DCNN2_LENGTH_IN  ;
                                end
                            end
                        end
                        else if (layer_done) begin
                                addr_sram_act_for_1 <= ADDR_DCNN2_SRAM_ACT_INIT;
                                addr_sram_act_for_2 <= ADDR_DCNN2_SRAM_ACT_INIT + CNN21_LENGTH_IN;
                                addr_sram_act_for_3 <= ADDR_DCNN2_SRAM_ACT_INIT + 2 * CNN21_LENGTH_IN;
                                addr_sram_act_for_4 <= ADDR_DCNN2_SRAM_ACT_INIT + 3 * CNN21_LENGTH_IN;
                                addr_sram_act_for_5 <= ADDR_DCNN2_SRAM_ACT_INIT + 4 * CNN21_LENGTH_IN;
                                addr_sram_act_for_6 <= ADDR_DCNN2_SRAM_ACT_INIT + 5 * CNN21_LENGTH_IN;
                                addr_sram_act_for_7 <= ADDR_DCNN2_SRAM_ACT_INIT + 6 * CNN21_LENGTH_IN;
                                addr_sram_act_for_8 <= ADDR_DCNN2_SRAM_ACT_INIT + 7 * CNN21_LENGTH_IN;
                                addr_sram_act_for_9 <= ADDR_DCNN2_SRAM_ACT_INIT + 8 * CNN21_LENGTH_IN;
                                addr_sram_act_for_10 <= ADDR_DCNN2_SRAM_ACT_INIT + 9 * CNN21_LENGTH_IN;
                                addr_sram_act_for_11 <= ADDR_DCNN2_SRAM_ACT_INIT + 10 * CNN21_LENGTH_IN;
                                addr_sram_act_for_12 <= ADDR_DCNN2_SRAM_ACT_INIT + 11 * CNN21_LENGTH_IN;
                                addr_sram_act_for_13 <= ADDR_DCNN2_SRAM_ACT_INIT + 12 * CNN21_LENGTH_IN;
                                addr_sram_act_for_14 <= ADDR_DCNN2_SRAM_ACT_INIT + 13 * CNN21_LENGTH_IN;
                                addr_sram_act_for_15 <= ADDR_DCNN2_SRAM_ACT_INIT + 14 * CNN21_LENGTH_IN;
                                addr_sram_act_for_16 <= ADDR_DCNN2_SRAM_ACT_INIT + 15 * CNN21_LENGTH_IN;
                                addr_sram_act_for_17 <= 0 ;
                                addr_sram_act_for_18 <= 0;
                                addr_sram_act_for_19 <= 0;
                                addr_sram_act_for_20 <= 0;
                                addr_sram_act_for_21 <= 0;
                                addr_sram_act_for_22 <= 0;
                                addr_sram_act_for_23 <= 0;
                                addr_sram_act_for_24 <= 0;
                                addr_sram_act_for_25 <= 0;
                                addr_sram_act_for_26 <= 0;
                                addr_sram_act_for_27 <= 0;
                                addr_sram_act_for_28 <= 0;
                                addr_sram_act_for_29 <= 0;
                                addr_sram_act_for_30 <= 0;
                                addr_sram_act_for_31 <= 0;
                                addr_sram_act_for_32 <= 0;                              
                        end   
                        else ;

                    end             
 
            end
            else;




            
        end
    end

    always @(*) begin
        if ((decoder_top_state == dcnn1)|(decoder_top_state == cnn11) | (decoder_top_state == cnn12) ) begin
        
            if ((conv_state == load_a) & (!addr_sram_act_re_end_1)) begin
                addr_sram_act_re = addr_sram_act_for_1;
                spad_a_we_en_1_32 = 1;
            end
            else if ((conv_state_mem[0] == load_a) & (!addr_sram_act_re_end_2)) begin
                addr_sram_act_re = addr_sram_act_for_2;
                spad_a_we_en_1_32 = 1<<1;
            end
            else if ((conv_state_mem[1] == load_a) & (!addr_sram_act_re_end_3)) begin
                addr_sram_act_re = addr_sram_act_for_3;
                spad_a_we_en_1_32 = 1<<2;
            end
            else if ((conv_state_mem[2] == load_a) & (!addr_sram_act_re_end_4)) begin
                addr_sram_act_re = addr_sram_act_for_4;
                spad_a_we_en_1_32 = 1<<3;
            end
            else if ((conv_state_mem[3] == load_a) & (!addr_sram_act_re_end_5)) begin
                addr_sram_act_re = addr_sram_act_for_5;
                spad_a_we_en_1_32 = 1<<4;
            end
            else if ((conv_state_mem[4] == load_a) & (!addr_sram_act_re_end_6)) begin
                addr_sram_act_re = addr_sram_act_for_6;
                spad_a_we_en_1_32 = 1<<5;
            end
            else if ((conv_state_mem[5] == load_a) & (!addr_sram_act_re_end_7)) begin
                addr_sram_act_re = addr_sram_act_for_7;
                spad_a_we_en_1_32 = 1<<6;
            end
            else if ((conv_state_mem[6] == load_a) & (!addr_sram_act_re_end_8)) begin
                addr_sram_act_re = addr_sram_act_for_8;
                spad_a_we_en_1_32 = 1<<7;
            end
            else if ((conv_state_mem[7] == load_a) & (!addr_sram_act_re_end_9)) begin
                addr_sram_act_re = addr_sram_act_for_9;
                spad_a_we_en_1_32 = 1<<8;
            end
            else if ((conv_state_mem[8] == load_a) & (!addr_sram_act_re_end_10)) begin
                addr_sram_act_re = addr_sram_act_for_10;
                spad_a_we_en_1_32 = 1<<9;
            end
            else if ((conv_state_mem[9] == load_a) & (!addr_sram_act_re_end_11)) begin
                addr_sram_act_re = addr_sram_act_for_11;
                spad_a_we_en_1_32 = 1<<10;
            end
            else if ((conv_state_mem[10] == load_a) & (!addr_sram_act_re_end_12)) begin
                addr_sram_act_re = addr_sram_act_for_12;
                spad_a_we_en_1_32 = 1<<11;
            end
            else if ((conv_state_mem[11] == load_a) & (!addr_sram_act_re_end_13)) begin
                addr_sram_act_re = addr_sram_act_for_13;
                spad_a_we_en_1_32 = 1<<12;
            end
            else if ((conv_state_mem[12] == load_a) & (!addr_sram_act_re_end_14)) begin
                addr_sram_act_re = addr_sram_act_for_14;
                spad_a_we_en_1_32 = 1<<13;
            end
            else if ((conv_state_mem[13] == load_a) & (!addr_sram_act_re_end_15)) begin
                addr_sram_act_re = addr_sram_act_for_15;
                spad_a_we_en_1_32 = 1<<14;
            end
            else if ((conv_state_mem[14] == load_a) & (!addr_sram_act_re_end_16)) begin
                addr_sram_act_re = addr_sram_act_for_16;
                spad_a_we_en_1_32 = 1<<15;
            end
            else if ((conv_state_mem[15] == load_a) & (!addr_sram_act_re_end_17)) begin
                addr_sram_act_re = addr_sram_act_for_17;
                spad_a_we_en_1_32 = 1<<16;
            end
            else if ((conv_state_mem[16] == load_a) & (!addr_sram_act_re_end_18)) begin
                addr_sram_act_re = addr_sram_act_for_18;
                spad_a_we_en_1_32 = 1<<17;
            end
            else if ((conv_state_mem[17] == load_a) & (!addr_sram_act_re_end_19)) begin
                addr_sram_act_re = addr_sram_act_for_19;
                spad_a_we_en_1_32 = 1<<18;
            end
            else if ((conv_state_mem[18] == load_a) & (!addr_sram_act_re_end_20)) begin
                addr_sram_act_re = addr_sram_act_for_20;
                spad_a_we_en_1_32 = 1<<19;
            end
            else if ((conv_state_mem[19] == load_a) & (!addr_sram_act_re_end_21)) begin
                addr_sram_act_re = addr_sram_act_for_21;
                spad_a_we_en_1_32 = 1<<20;
            end
            else if ((conv_state_mem[20] == load_a) & (!addr_sram_act_re_end_22)) begin
                addr_sram_act_re = addr_sram_act_for_22;
                spad_a_we_en_1_32 = 1<<21;
            end
            else if ((conv_state_mem[21] == load_a) & (!addr_sram_act_re_end_23)) begin
                addr_sram_act_re = addr_sram_act_for_23;
                spad_a_we_en_1_32 = 1<<22;
            end
            else if ((conv_state_mem[22] == load_a) & (!addr_sram_act_re_end_24)) begin
                addr_sram_act_re = addr_sram_act_for_24;
                spad_a_we_en_1_32 = 1<<23;
            end
            else if ((conv_state_mem[23] == load_a) & (!addr_sram_act_re_end_25)) begin
                addr_sram_act_re = addr_sram_act_for_25;
                spad_a_we_en_1_32 = 1<<24;
            end
            else if ((conv_state_mem[24] == load_a) & (!addr_sram_act_re_end_26)) begin
                addr_sram_act_re = addr_sram_act_for_26;
                spad_a_we_en_1_32 = 1<<25;
            end
            else if ((conv_state_mem[25] == load_a) & (!addr_sram_act_re_end_27)) begin
                addr_sram_act_re = addr_sram_act_for_27;
                spad_a_we_en_1_32 = 1<<26;
            end
            else if ((conv_state_mem[26] == load_a) & (!addr_sram_act_re_end_28)) begin
                addr_sram_act_re = addr_sram_act_for_28;
                spad_a_we_en_1_32 = 1<<27;
            end
            else if ((conv_state_mem[27] == load_a) & (!addr_sram_act_re_end_29)) begin
                addr_sram_act_re = addr_sram_act_for_29;
                spad_a_we_en_1_32 = 1<<28;
            end
            else if ((conv_state_mem[28] == load_a) & (!addr_sram_act_re_end_30)) begin
                addr_sram_act_re = addr_sram_act_for_30;
                spad_a_we_en_1_32 = 1<<29;
            end
            else if ((conv_state_mem[29] == load_a) & (!addr_sram_act_re_end_31)) begin
                addr_sram_act_re = addr_sram_act_for_31;
                spad_a_we_en_1_32 = 1<<30;
            end
            else if ((conv_state_mem[30] == load_a) & (!addr_sram_act_re_end_32)) begin
                addr_sram_act_re = addr_sram_act_for_32;
                spad_a_we_en_1_32 = 1<<31;
            end
            else begin
                addr_sram_act_re = 0;
                spad_a_we_en_1_32 = 0;
            end 
        end
        else if (decoder_top_state == dcnn2) begin
            if ((conv_state == load_a) & (!addr_sram_act_re_end_1)) begin
                addr_sram_act_re = addr_sram_act_for_1;
                spad_a_we_en_1_32 = 1 |  1<<16;
            end
            else if ((conv_state_mem[0] == load_a) & (!addr_sram_act_re_end_2)) begin
                addr_sram_act_re = addr_sram_act_for_2;
                spad_a_we_en_1_32 = 1<<1 |  1<<17;
            end
            else if ((conv_state_mem[1] == load_a) & (!addr_sram_act_re_end_3)) begin
                addr_sram_act_re = addr_sram_act_for_3;
                spad_a_we_en_1_32 = 1<<2 |  1<<18;
            end
            else if ((conv_state_mem[2] == load_a) & (!addr_sram_act_re_end_4)) begin
                addr_sram_act_re = addr_sram_act_for_4;
                spad_a_we_en_1_32 = 1<<3  |  1<<19;
            end
            else if ((conv_state_mem[3] == load_a) & (!addr_sram_act_re_end_5)) begin
                addr_sram_act_re = addr_sram_act_for_5;
                spad_a_we_en_1_32 = 1<<4 |  1<<20;
            end
            else if ((conv_state_mem[4] == load_a) & (!addr_sram_act_re_end_6)) begin
                addr_sram_act_re = addr_sram_act_for_6;
                spad_a_we_en_1_32 = 1<<5 |  1<<21;
            end
            else if ((conv_state_mem[5] == load_a) & (!addr_sram_act_re_end_7)) begin
                addr_sram_act_re = addr_sram_act_for_7;
                spad_a_we_en_1_32 = 1<<6 |  1<<22;
            end
            else if ((conv_state_mem[6] == load_a) & (!addr_sram_act_re_end_8)) begin
                addr_sram_act_re = addr_sram_act_for_8;
                spad_a_we_en_1_32 = 1<<7 |  1<<23;
            end
            else if ((conv_state_mem[7] == load_a) & (!addr_sram_act_re_end_9)) begin
                addr_sram_act_re = addr_sram_act_for_9;
                spad_a_we_en_1_32 = 1<<8 |  1<<24;
            end
            else if ((conv_state_mem[8] == load_a) & (!addr_sram_act_re_end_10)) begin
                addr_sram_act_re = addr_sram_act_for_10;
                spad_a_we_en_1_32 = 1<<9 |  1<<25;
            end
            else if ((conv_state_mem[9] == load_a) & (!addr_sram_act_re_end_11)) begin
                addr_sram_act_re = addr_sram_act_for_11;
                spad_a_we_en_1_32 = 1<<10 |  1<<26;
            end
            else if ((conv_state_mem[10] == load_a) & (!addr_sram_act_re_end_12)) begin
                addr_sram_act_re = addr_sram_act_for_12;
                spad_a_we_en_1_32 = 1<<11 |  1<<27;
            end
            else if ((conv_state_mem[11] == load_a) & (!addr_sram_act_re_end_13)) begin
                addr_sram_act_re = addr_sram_act_for_13;
                spad_a_we_en_1_32 = 1<<12 |  1<<28;
            end
            else if ((conv_state_mem[12] == load_a) & (!addr_sram_act_re_end_14)) begin
                addr_sram_act_re = addr_sram_act_for_14;
                spad_a_we_en_1_32 = 1<<13 |  1<<29;
            end
            else if ((conv_state_mem[13] == load_a) & (!addr_sram_act_re_end_15)) begin
                addr_sram_act_re = addr_sram_act_for_15;
                spad_a_we_en_1_32 = 1<<14 |  1<<30;
            end
            else if ((conv_state_mem[14] == load_a) & (!addr_sram_act_re_end_16)) begin
                addr_sram_act_re = addr_sram_act_for_16;
                spad_a_we_en_1_32 = 1<<15 |  1<<31;
            end
            else begin
                addr_sram_act_re = 0;
                spad_a_we_en_1_32 = 0;                   
            end            
        end
        else if (decoder_top_state_8) begin
            if ((conv_state == load_a) & (!addr_sram_act_re_end_1)) begin
                addr_sram_act_re = addr_sram_act_for_1;
                spad_a_we_en_1_32 = 1 |  1<<16 | 1<<8 | 1<<24;
            end
            else if ((conv_state_mem[0] == load_a) & (!addr_sram_act_re_end_2)) begin
                addr_sram_act_re = addr_sram_act_for_2;
                spad_a_we_en_1_32 = 1<<1 |  1<<17 | 1<<9 |  1<<25;
            end
            else if ((conv_state_mem[1] == load_a) & (!addr_sram_act_re_end_3)) begin
                addr_sram_act_re = addr_sram_act_for_3;
                spad_a_we_en_1_32 = 1<<2 |  1<<18 | 1<<10 |  1<<26;
            end
            else if ((conv_state_mem[2] == load_a) & (!addr_sram_act_re_end_4)) begin
                addr_sram_act_re = addr_sram_act_for_4;
                spad_a_we_en_1_32 = 1<<3  |  1<<19 | 1<<11 |  1<<27;
            end
            else if ((conv_state_mem[3] == load_a) & (!addr_sram_act_re_end_5)) begin
                addr_sram_act_re = addr_sram_act_for_5;
                spad_a_we_en_1_32 = 1<<4 |  1<<20 | 1<<12 |  1<<28;
            end
            else if ((conv_state_mem[4] == load_a) & (!addr_sram_act_re_end_6)) begin
                addr_sram_act_re = addr_sram_act_for_6;
                spad_a_we_en_1_32 = 1<<5 |  1<<21 | 1<<13 |  1<<29;
            end
            else if ((conv_state_mem[5] == load_a) & (!addr_sram_act_re_end_7)) begin
                addr_sram_act_re = addr_sram_act_for_7;
                spad_a_we_en_1_32 = 1<<6 |  1<<22 | 1<<14 |  1<<30;
            end
            else if ((conv_state_mem[6] == load_a) & (!addr_sram_act_re_end_8)) begin
                addr_sram_act_re = addr_sram_act_for_8;
                spad_a_we_en_1_32 = 1<<7 |  1<<23 | 1<<15 |  1<<31;
            end
            else begin
                addr_sram_act_re = 0;
                spad_a_we_en_1_32 = 0;
            end
            
        end
        else begin
            addr_sram_act_re = 0;
            spad_a_we_en_1_32 = 0;            
        end
    end


always @(negedge sclk or negedge rst_n) begin
    if (!rst_n) begin
        spad_a_addr_we <= 0;
    end
    else begin

        if (sram_act_we) begin
            spad_a_addr_we  <= spad_a_addr_we;    
        end
        else begin
            if ((conv_state == load_a) & (!addr_sram_act_re_end_1))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[0] == load_a) & (!addr_sram_act_re_end_2))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[1] == load_a) & (!addr_sram_act_re_end_3))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[2] == load_a) & (!addr_sram_act_re_end_4))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[3] == load_a) & (!addr_sram_act_re_end_5))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[4] == load_a) & (!addr_sram_act_re_end_6))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[5] == load_a) & (!addr_sram_act_re_end_7))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[6] == load_a) & (!addr_sram_act_re_end_8))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[7] == load_a) & (!addr_sram_act_re_end_9))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[8] == load_a) & (!addr_sram_act_re_end_10))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[9] == load_a) & (!addr_sram_act_re_end_11))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[10] == load_a) & (!addr_sram_act_re_end_12))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[11] == load_a) & (!addr_sram_act_re_end_13))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[12] == load_a) & (!addr_sram_act_re_end_14))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[13] == load_a) & (!addr_sram_act_re_end_15))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[14] == load_a) & (!addr_sram_act_re_end_16))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[15] == load_a) & (!addr_sram_act_re_end_17))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[16] == load_a) & (!addr_sram_act_re_end_18))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[17] == load_a) & (!addr_sram_act_re_end_19))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[18] == load_a) & (!addr_sram_act_re_end_20))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[19] == load_a) & (!addr_sram_act_re_end_21))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[20] == load_a) & (!addr_sram_act_re_end_22))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[21] == load_a) & (!addr_sram_act_re_end_23))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[22] == load_a) & (!addr_sram_act_re_end_24))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[23] == load_a) & (!addr_sram_act_re_end_25))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[24] == load_a) & (!addr_sram_act_re_end_26))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[25] == load_a) & (!addr_sram_act_re_end_27))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[26] == load_a) & (!addr_sram_act_re_end_28))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[27] == load_a) & (!addr_sram_act_re_end_29))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[28] == load_a) & (!addr_sram_act_re_end_30))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[29] == load_a) & (!addr_sram_act_re_end_31))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else if ((conv_state_mem[30] == load_a) & (!addr_sram_act_re_end_32))
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            else
                spad_a_addr_we <= 0;
        end
 
    end
end



always @(*) begin
    if (decoder_top_state_dcnn) begin

        if ((conv_state == load_a) & !addr_sram_act_re_end_1) begin
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
            case (padding_crl)
            0:begin
                if (~is_odd) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad1_a_data_sram_in = {{(INPUT_DW - SRAM8_DW){sram_act_dout[SRAM8_DW-1]}} , sram_act_dout};
                    else spad1_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad1_a_data_sram_in = {{(INPUT_DW - SRAM8_DW){sram_act_dout[SRAM8_DW-1]}} , sram_act_dout};
                    else spad1_a_data_sram_in = 0;                    
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad1_a_data_sram_in = {{(INPUT_DW - SRAM8_DW){sram_act_dout[SRAM8_DW-1]}} , sram_act_dout};
                else spad1_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad1_a_data_sram_in = {{(INPUT_DW - SRAM8_DW){sram_act_dout[SRAM8_DW-1]}} , sram_act_dout};
                else spad1_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad1_a_data_sram_in = {{(INPUT_DW - SRAM8_DW){sram_act_dout[SRAM8_DW-1]}} , sram_act_dout};
                else spad1_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad1_a_data_sram_in = {{(INPUT_DW - SRAM8_DW){sram_act_dout[SRAM8_DW-1]}} , sram_act_dout};
                else spad1_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad1_a_data_sram_in = {{(INPUT_DW - SRAM8_DW){sram_act_dout[SRAM8_DW-1]}} , sram_act_dout};
                else spad1_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad1_a_data_sram_in = {{(INPUT_DW - SRAM8_DW){sram_act_dout[SRAM8_DW-1]}} , sram_act_dout};
                else spad1_a_data_sram_in = 0;
            end
            default:spad1_a_data_sram_in = 0;
            endcase
            
        end 
        else if ((conv_state_mem[0] == load_a) & !addr_sram_act_re_end_2) begin
            spad1_a_data_sram_in = 0;        
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
            case (padding_crl_mem[0])
            0:begin
                if (~is_odd_mem[0]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad2_a_data_sram_in = sram_act_dout;
                    else spad2_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad2_a_data_sram_in = sram_act_dout;
                    else spad2_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad2_a_data_sram_in = sram_act_dout;
                else spad2_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad2_a_data_sram_in = sram_act_dout;
                else spad2_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad2_a_data_sram_in = sram_act_dout;
                else spad2_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad2_a_data_sram_in = sram_act_dout;
                else spad2_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad2_a_data_sram_in = sram_act_dout;
                else spad2_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad2_a_data_sram_in = sram_act_dout;
                else spad2_a_data_sram_in = 0;
            end
            default:spad2_a_data_sram_in = 0;
            endcase
            
        end
        else if ((conv_state_mem[1] == load_a) & !addr_sram_act_re_end_3) begin
            spad1_a_data_sram_in = 0;        
            spad2_a_data_sram_in = 0; 
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
            case (padding_crl_mem[1])
            0:begin
                if (~is_odd_mem[1]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad3_a_data_sram_in = sram_act_dout;
                    else spad3_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad3_a_data_sram_in = sram_act_dout;
                    else spad3_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad3_a_data_sram_in = sram_act_dout;
                else spad3_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad3_a_data_sram_in = sram_act_dout;
                else spad3_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad3_a_data_sram_in = sram_act_dout;
                else spad3_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad3_a_data_sram_in = sram_act_dout;
                else spad3_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad3_a_data_sram_in = sram_act_dout;
                else spad3_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad3_a_data_sram_in = sram_act_dout;
                else spad3_a_data_sram_in = 0;
            end
            default:spad3_a_data_sram_in = 0;
            endcase
            
        end
        else if ((conv_state_mem[2] == load_a) & !addr_sram_act_re_end_4) begin
            spad1_a_data_sram_in = 0;        
            spad2_a_data_sram_in = 0; 
            spad3_a_data_sram_in = 0; 
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
            case (padding_crl_mem[2])
            0:begin
                if (~is_odd_mem[2]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad4_a_data_sram_in = sram_act_dout;
                    else spad4_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad4_a_data_sram_in = sram_act_dout;
                    else spad4_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad4_a_data_sram_in = sram_act_dout;
                else spad4_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad4_a_data_sram_in = sram_act_dout;
                else spad4_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad4_a_data_sram_in = sram_act_dout;
                else spad4_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad4_a_data_sram_in = sram_act_dout;
                else spad4_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad4_a_data_sram_in = sram_act_dout;
                else spad4_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad4_a_data_sram_in = sram_act_dout;
                else spad4_a_data_sram_in = 0;
            end
            default:spad4_a_data_sram_in = 0;
            endcase
            
        end
        else if ((conv_state_mem[3] == load_a) & !addr_sram_act_re_end_5) begin
            spad1_a_data_sram_in = 0;        
            spad2_a_data_sram_in = 0; 
            spad3_a_data_sram_in = 0; 
            spad4_a_data_sram_in = 0; 
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
            case (padding_crl_mem[3])
            0:begin
                if (~is_odd_mem[3]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad5_a_data_sram_in = sram_act_dout;
                    else spad5_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad5_a_data_sram_in = sram_act_dout;
                    else spad5_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad5_a_data_sram_in = sram_act_dout;
                else spad5_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad5_a_data_sram_in = sram_act_dout;
                else spad5_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad5_a_data_sram_in = sram_act_dout;
                else spad5_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad5_a_data_sram_in = sram_act_dout;
                else spad5_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad5_a_data_sram_in = sram_act_dout;
                else spad5_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad5_a_data_sram_in = sram_act_dout;
                else spad5_a_data_sram_in = 0;
            end
            default:spad5_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[4] == load_a) & !addr_sram_act_re_end_6) begin
            spad1_a_data_sram_in = 0;        
            spad2_a_data_sram_in = 0; 
            spad3_a_data_sram_in = 0; 
            spad4_a_data_sram_in = 0; 
            spad5_a_data_sram_in = 0; 
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
            case (padding_crl_mem[4])
            0:begin
                if (~is_odd_mem[4]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad6_a_data_sram_in = sram_act_dout;
                    else spad6_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad6_a_data_sram_in = sram_act_dout;
                    else spad6_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad6_a_data_sram_in = sram_act_dout;
                else spad6_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad6_a_data_sram_in = sram_act_dout;
                else spad6_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad6_a_data_sram_in = sram_act_dout;
                else spad6_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad6_a_data_sram_in = sram_act_dout;
                else spad6_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad6_a_data_sram_in = sram_act_dout;
                else spad6_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad6_a_data_sram_in = sram_act_dout;
                else spad6_a_data_sram_in = 0;
            end
            default:spad6_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[5] == load_a) & !addr_sram_act_re_end_7) begin
            spad1_a_data_sram_in = 0;        
            spad2_a_data_sram_in = 0; 
            spad3_a_data_sram_in = 0; 
            spad4_a_data_sram_in = 0; 
            spad5_a_data_sram_in = 0; 
            spad6_a_data_sram_in = 0; 
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
            case (padding_crl_mem[5])
            0:begin
                if (~is_odd_mem[5]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad7_a_data_sram_in = sram_act_dout;
                    else spad7_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad7_a_data_sram_in = sram_act_dout;
                    else spad7_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad7_a_data_sram_in = sram_act_dout;
                else spad7_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad7_a_data_sram_in = sram_act_dout;
                else spad7_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad7_a_data_sram_in = sram_act_dout;
                else spad7_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad7_a_data_sram_in = sram_act_dout;
                else spad7_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad7_a_data_sram_in = sram_act_dout;
                else spad7_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad7_a_data_sram_in = sram_act_dout;
                else spad7_a_data_sram_in = 0;
            end
            default:spad7_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[6] == load_a) & !addr_sram_act_re_end_8) begin
            spad1_a_data_sram_in = 0;        
            spad2_a_data_sram_in = 0; 
            spad3_a_data_sram_in = 0; 
            spad4_a_data_sram_in = 0; 
            spad5_a_data_sram_in = 0; 
            spad6_a_data_sram_in = 0; 
            spad7_a_data_sram_in = 0; 
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
            case (padding_crl_mem[6])
            0:begin
                if (~is_odd_mem[6]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad8_a_data_sram_in = sram_act_dout;
                    else spad8_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad8_a_data_sram_in = sram_act_dout;
                    else spad8_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad8_a_data_sram_in = sram_act_dout;
                else spad8_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad8_a_data_sram_in = sram_act_dout;
                else spad8_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad8_a_data_sram_in = sram_act_dout;
                else spad8_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad8_a_data_sram_in = sram_act_dout;
                else spad8_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad8_a_data_sram_in = sram_act_dout;
                else spad8_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad8_a_data_sram_in = sram_act_dout;
                else spad8_a_data_sram_in = 0;
            end
            default:spad8_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[7] == load_a) & !addr_sram_act_re_end_9) begin
            spad1_a_data_sram_in = 0;        
            spad2_a_data_sram_in = 0; 
            spad3_a_data_sram_in = 0; 
            spad4_a_data_sram_in = 0; 
            spad5_a_data_sram_in = 0; 
            spad6_a_data_sram_in = 0; 
            spad7_a_data_sram_in = 0; 
            spad8_a_data_sram_in = 0; 
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
            case (padding_crl_mem[7])
            0:begin
                if (~is_odd_mem[7]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad9_a_data_sram_in = sram_act_dout;
                    else spad9_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad9_a_data_sram_in = sram_act_dout;
                    else spad9_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad9_a_data_sram_in = sram_act_dout;
                else spad9_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad9_a_data_sram_in = sram_act_dout;
                else spad9_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad9_a_data_sram_in = sram_act_dout;
                else spad9_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad9_a_data_sram_in = sram_act_dout;
                else spad9_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad9_a_data_sram_in = sram_act_dout;
                else spad9_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad9_a_data_sram_in = sram_act_dout;
                else spad9_a_data_sram_in = 0;
            end
            default:spad9_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[8] == load_a) & !addr_sram_act_re_end_10) begin
            spad1_a_data_sram_in = 0;        
            spad2_a_data_sram_in = 0; 
            spad3_a_data_sram_in = 0; 
            spad4_a_data_sram_in = 0; 
            spad5_a_data_sram_in = 0; 
            spad6_a_data_sram_in = 0; 
            spad7_a_data_sram_in = 0; 
            spad8_a_data_sram_in = 0; 
            spad9_a_data_sram_in = 0; 
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
            case (padding_crl_mem[8])
            0:begin
                if (~is_odd_mem[8]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad10_a_data_sram_in = sram_act_dout;
                    else spad10_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad10_a_data_sram_in = sram_act_dout;
                    else spad10_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad10_a_data_sram_in = sram_act_dout;
                else spad10_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad10_a_data_sram_in = sram_act_dout;
                else spad10_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad10_a_data_sram_in = sram_act_dout;
                else spad10_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad10_a_data_sram_in = sram_act_dout;
                else spad10_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad10_a_data_sram_in = sram_act_dout;
                else spad10_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad10_a_data_sram_in = sram_act_dout;
                else spad10_a_data_sram_in = 0;
            end
            default:spad10_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[9] == load_a) & !addr_sram_act_re_end_11) begin
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
            case (padding_crl_mem[9])
            0:begin
                if (~is_odd_mem[9]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad11_a_data_sram_in = sram_act_dout;
                    else spad11_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad11_a_data_sram_in = sram_act_dout;
                    else spad11_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad11_a_data_sram_in = sram_act_dout;
                else spad11_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad11_a_data_sram_in = sram_act_dout;
                else spad11_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad11_a_data_sram_in = sram_act_dout;
                else spad11_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad11_a_data_sram_in = sram_act_dout;
                else spad11_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad11_a_data_sram_in = sram_act_dout;
                else spad11_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad11_a_data_sram_in = sram_act_dout;
                else spad11_a_data_sram_in = 0;
            end
            default:spad11_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[10] == load_a) & !addr_sram_act_re_end_12) begin
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
            case (padding_crl_mem[10])
            0:begin
                if (~is_odd_mem[10]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad12_a_data_sram_in = sram_act_dout;
                    else spad12_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad12_a_data_sram_in = sram_act_dout;
                    else spad12_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad12_a_data_sram_in = sram_act_dout;
                else spad12_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad12_a_data_sram_in = sram_act_dout;
                else spad12_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad12_a_data_sram_in = sram_act_dout;
                else spad12_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad12_a_data_sram_in = sram_act_dout;
                else spad12_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad12_a_data_sram_in = sram_act_dout;
                else spad12_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad12_a_data_sram_in = sram_act_dout;
                else spad12_a_data_sram_in = 0;
            end
            default:spad12_a_data_sram_in = 0;
            endcase
            
        end
        else if ((conv_state_mem[11] == load_a) & !addr_sram_act_re_end_13) begin
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
            case (padding_crl_mem[11])
            0:begin
                if (~is_odd_mem[11]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad13_a_data_sram_in = sram_act_dout;
                    else spad13_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad13_a_data_sram_in = sram_act_dout;
                    else spad13_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad13_a_data_sram_in = sram_act_dout;
                else spad13_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad13_a_data_sram_in = sram_act_dout;
                else spad13_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad13_a_data_sram_in = sram_act_dout;
                else spad13_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad13_a_data_sram_in = sram_act_dout;
                else spad13_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad13_a_data_sram_in = sram_act_dout;
                else spad13_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad13_a_data_sram_in = sram_act_dout;
                else spad13_a_data_sram_in = 0;
            end
            default:spad13_a_data_sram_in = 0;
            endcase
            
        end
        else if ((conv_state_mem[12] == load_a) & !addr_sram_act_re_end_14) begin
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
            case (padding_crl_mem[12])
            0:begin
                if (~is_odd_mem[12]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad14_a_data_sram_in = sram_act_dout;
                    else spad14_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad14_a_data_sram_in = sram_act_dout;
                    else spad14_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad14_a_data_sram_in = sram_act_dout;
                else spad14_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad14_a_data_sram_in = sram_act_dout;
                else spad14_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad14_a_data_sram_in = sram_act_dout;
                else spad14_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad14_a_data_sram_in = sram_act_dout;
                else spad14_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad14_a_data_sram_in = sram_act_dout;
                else spad14_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad14_a_data_sram_in = sram_act_dout;
                else spad14_a_data_sram_in = 0;
            end
            default:spad14_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[13] == load_a) & !addr_sram_act_re_end_15) begin
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
            case (padding_crl_mem[13])
            0:begin
                if (~is_odd_mem[13]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad15_a_data_sram_in = sram_act_dout;
                    else spad15_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad15_a_data_sram_in = sram_act_dout;
                    else spad15_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad15_a_data_sram_in = sram_act_dout;
                else spad15_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad15_a_data_sram_in = sram_act_dout;
                else spad15_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad15_a_data_sram_in = sram_act_dout;
                else spad15_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad15_a_data_sram_in = sram_act_dout;
                else spad15_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad15_a_data_sram_in = sram_act_dout;
                else spad15_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad15_a_data_sram_in = sram_act_dout;
                else spad15_a_data_sram_in = 0;
            end
            default:spad15_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[14] == load_a) & !addr_sram_act_re_end_16) begin
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
            case (padding_crl_mem[14])
            0:begin
                if (~is_odd_mem[14]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad16_a_data_sram_in = sram_act_dout;
                    else spad16_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad16_a_data_sram_in = sram_act_dout;
                    else spad16_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad16_a_data_sram_in = sram_act_dout;
                else spad16_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad16_a_data_sram_in = sram_act_dout;
                else spad16_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad16_a_data_sram_in = sram_act_dout;
                else spad16_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad16_a_data_sram_in = sram_act_dout;
                else spad16_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad16_a_data_sram_in = sram_act_dout;
                else spad16_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad16_a_data_sram_in = sram_act_dout;
                else spad16_a_data_sram_in = 0;
            end
            default:spad16_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[15] == load_a) & !addr_sram_act_re_end_17) begin
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
            case (padding_crl_mem[15])
            0:begin
                if (~is_odd_mem[15]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad17_a_data_sram_in = sram_act_dout;
                    else spad17_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad17_a_data_sram_in = sram_act_dout;
                    else spad17_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad17_a_data_sram_in = sram_act_dout;
                else spad17_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad17_a_data_sram_in = sram_act_dout;
                else spad17_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad17_a_data_sram_in = sram_act_dout;
                else spad17_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad17_a_data_sram_in = sram_act_dout;
                else spad17_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad17_a_data_sram_in = sram_act_dout;
                else spad17_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad17_a_data_sram_in = sram_act_dout;
                else spad17_a_data_sram_in = 0;
            end
            default:spad17_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[16] == load_a) & !addr_sram_act_re_end_18) begin
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
            // spad18_a_data_sram_in = 0; 
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
            case (padding_crl_mem[16])
            0:begin
                if (~is_odd_mem[16]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad18_a_data_sram_in = sram_act_dout;
                    else spad18_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad18_a_data_sram_in = sram_act_dout;
                    else spad18_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad18_a_data_sram_in = sram_act_dout;
                else spad18_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad18_a_data_sram_in = sram_act_dout;
                else spad18_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad18_a_data_sram_in = sram_act_dout;
                else spad18_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad18_a_data_sram_in = sram_act_dout;
                else spad18_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad18_a_data_sram_in = sram_act_dout;
                else spad18_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad18_a_data_sram_in = sram_act_dout;
                else spad18_a_data_sram_in = 0;
            end
            default:spad18_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[17] == load_a) & !addr_sram_act_re_end_19) begin
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
            // spad19_a_data_sram_in = 0; 
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
            case (padding_crl_mem[17])
            0:begin
                if (~is_odd_mem[17]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad19_a_data_sram_in = sram_act_dout;
                    else spad19_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad19_a_data_sram_in = sram_act_dout;
                    else spad19_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad19_a_data_sram_in = sram_act_dout;
                else spad19_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad19_a_data_sram_in = sram_act_dout;
                else spad19_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad19_a_data_sram_in = sram_act_dout;
                else spad19_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad19_a_data_sram_in = sram_act_dout;
                else spad19_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad19_a_data_sram_in = sram_act_dout;
                else spad19_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad19_a_data_sram_in = sram_act_dout;
                else spad19_a_data_sram_in = 0;
            end
            default:spad19_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[18] == load_a) & !addr_sram_act_re_end_20) begin
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
            // spad20_a_data_sram_in = 0; 
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
            case (padding_crl_mem[18])
            0:begin
                if (~is_odd_mem[18]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad20_a_data_sram_in = sram_act_dout;
                    else spad20_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad20_a_data_sram_in = sram_act_dout;
                    else spad20_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad20_a_data_sram_in = sram_act_dout;
                else spad20_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad20_a_data_sram_in = sram_act_dout;
                else spad20_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad20_a_data_sram_in = sram_act_dout;
                else spad20_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad20_a_data_sram_in = sram_act_dout;
                else spad20_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad20_a_data_sram_in = sram_act_dout;
                else spad20_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad20_a_data_sram_in = sram_act_dout;
                else spad20_a_data_sram_in = 0;
            end
            default:spad20_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[19] == load_a) & !addr_sram_act_re_end_21) begin
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
            // spad21_a_data_sram_in = 0; 
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
            case (padding_crl_mem[19])
            0:begin
                if (~is_odd_mem[19]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad21_a_data_sram_in = sram_act_dout;
                    else spad21_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad21_a_data_sram_in = sram_act_dout;
                    else spad21_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad21_a_data_sram_in = sram_act_dout;
                else spad21_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad21_a_data_sram_in = sram_act_dout;
                else spad21_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad21_a_data_sram_in = sram_act_dout;
                else spad21_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad21_a_data_sram_in = sram_act_dout;
                else spad21_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad21_a_data_sram_in = sram_act_dout;
                else spad21_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad21_a_data_sram_in = sram_act_dout;
                else spad21_a_data_sram_in = 0;
            end
            default:spad21_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[20] == load_a) & !addr_sram_act_re_end_22) begin
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
            // spad22_a_data_sram_in = 0; 
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
            case (padding_crl_mem[20])
            0:begin
                if (~is_odd_mem[20]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad22_a_data_sram_in = sram_act_dout;
                    else spad22_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad22_a_data_sram_in = sram_act_dout;
                    else spad22_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad22_a_data_sram_in = sram_act_dout;
                else spad22_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad22_a_data_sram_in = sram_act_dout;
                else spad22_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad22_a_data_sram_in = sram_act_dout;
                else spad22_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad22_a_data_sram_in = sram_act_dout;
                else spad22_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad22_a_data_sram_in = sram_act_dout;
                else spad22_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad22_a_data_sram_in = sram_act_dout;
                else spad22_a_data_sram_in = 0;
            end
            default:spad22_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[21] == load_a) & !addr_sram_act_re_end_23) begin
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
            // spad23_a_data_sram_in = 0; 
            spad24_a_data_sram_in = 0;
            spad25_a_data_sram_in = 0;
            spad26_a_data_sram_in = 0; 
            spad27_a_data_sram_in = 0; 
            spad28_a_data_sram_in = 0; 
            spad29_a_data_sram_in = 0; 
            spad30_a_data_sram_in = 0; 
            spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;
            case (padding_crl_mem[21])
            0:begin
                if (~is_odd_mem[21]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad23_a_data_sram_in = sram_act_dout;
                    else spad23_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad23_a_data_sram_in = sram_act_dout;
                    else spad23_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad23_a_data_sram_in = sram_act_dout;
                else spad23_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad23_a_data_sram_in = sram_act_dout;
                else spad23_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad23_a_data_sram_in = sram_act_dout;
                else spad23_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad23_a_data_sram_in = sram_act_dout;
                else spad23_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad23_a_data_sram_in = sram_act_dout;
                else spad23_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad23_a_data_sram_in = sram_act_dout;
                else spad23_a_data_sram_in = 0;
            end
            default:spad23_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[22] == load_a) & !addr_sram_act_re_end_24) begin
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
            // spad24_a_data_sram_in = 0;
            spad25_a_data_sram_in = 0;
            spad26_a_data_sram_in = 0; 
            spad27_a_data_sram_in = 0; 
            spad28_a_data_sram_in = 0; 
            spad29_a_data_sram_in = 0; 
            spad30_a_data_sram_in = 0; 
            spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;
            case (padding_crl_mem[22])
            0:begin
                if (~is_odd_mem[22]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad24_a_data_sram_in = sram_act_dout;
                    else spad24_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad24_a_data_sram_in = sram_act_dout;
                    else spad24_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad24_a_data_sram_in = sram_act_dout;
                else spad24_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad24_a_data_sram_in = sram_act_dout;
                else spad24_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad24_a_data_sram_in = sram_act_dout;
                else spad24_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad24_a_data_sram_in = sram_act_dout;
                else spad24_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad24_a_data_sram_in = sram_act_dout;
                else spad24_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad24_a_data_sram_in = sram_act_dout;
                else spad24_a_data_sram_in = 0;
            end
            default:spad24_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[23] == load_a) & !addr_sram_act_re_end_25) begin
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
            // spad25_a_data_sram_in = 0;
            spad26_a_data_sram_in = 0; 
            spad27_a_data_sram_in = 0; 
            spad28_a_data_sram_in = 0; 
            spad29_a_data_sram_in = 0; 
            spad30_a_data_sram_in = 0; 
            spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;
            case (padding_crl_mem[23])
            0:begin
                if (~is_odd_mem[23]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad25_a_data_sram_in = sram_act_dout;
                    else spad25_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad25_a_data_sram_in = sram_act_dout;
                    else spad25_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad25_a_data_sram_in = sram_act_dout;
                else spad25_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad25_a_data_sram_in = sram_act_dout;
                else spad25_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad25_a_data_sram_in = sram_act_dout;
                else spad25_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad25_a_data_sram_in = sram_act_dout;
                else spad25_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad25_a_data_sram_in = sram_act_dout;
                else spad25_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad25_a_data_sram_in = sram_act_dout;
                else spad25_a_data_sram_in = 0;
            end
            default:spad25_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[24] == load_a) & !addr_sram_act_re_end_26) begin
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
            // spad26_a_data_sram_in = 0; 
            spad27_a_data_sram_in = 0; 
            spad28_a_data_sram_in = 0; 
            spad29_a_data_sram_in = 0; 
            spad30_a_data_sram_in = 0; 
            spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;
            case (padding_crl_mem[24])
            0:begin
                if (~is_odd_mem[24]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad26_a_data_sram_in = sram_act_dout;
                    else spad26_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad26_a_data_sram_in = sram_act_dout;
                    else spad26_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad26_a_data_sram_in = sram_act_dout;
                else spad26_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad26_a_data_sram_in = sram_act_dout;
                else spad26_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad26_a_data_sram_in = sram_act_dout;
                else spad26_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad26_a_data_sram_in = sram_act_dout;
                else spad26_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad26_a_data_sram_in = sram_act_dout;
                else spad26_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad26_a_data_sram_in = sram_act_dout;
                else spad26_a_data_sram_in = 0;
            end
            default:spad26_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[25] == load_a) & !addr_sram_act_re_end_27) begin
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
            // spad27_a_data_sram_in = 0; 
            spad28_a_data_sram_in = 0; 
            spad29_a_data_sram_in = 0; 
            spad30_a_data_sram_in = 0; 
            spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;
            case (padding_crl_mem[25])
            0:begin
                if (~is_odd_mem[25]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad27_a_data_sram_in = sram_act_dout;
                    else spad27_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad27_a_data_sram_in = sram_act_dout;
                    else spad27_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad27_a_data_sram_in = sram_act_dout;
                else spad27_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad27_a_data_sram_in = sram_act_dout;
                else spad27_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad27_a_data_sram_in = sram_act_dout;
                else spad27_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad27_a_data_sram_in = sram_act_dout;
                else spad27_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad27_a_data_sram_in = sram_act_dout;
                else spad27_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad27_a_data_sram_in = sram_act_dout;
                else spad27_a_data_sram_in = 0;
            end
            default:spad27_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[26] == load_a) & !addr_sram_act_re_end_28) begin
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
            // spad28_a_data_sram_in = 0; 
            spad29_a_data_sram_in = 0; 
            spad30_a_data_sram_in = 0; 
            spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;
            case (padding_crl_mem[26])
            0:begin
                if (~is_odd_mem[26]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad28_a_data_sram_in = sram_act_dout;
                    else spad28_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad28_a_data_sram_in = sram_act_dout;
                    else spad28_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad28_a_data_sram_in = sram_act_dout;
                else spad28_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad28_a_data_sram_in = sram_act_dout;
                else spad28_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad28_a_data_sram_in = sram_act_dout;
                else spad28_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad28_a_data_sram_in = sram_act_dout;
                else spad28_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad28_a_data_sram_in = sram_act_dout;
                else spad28_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad28_a_data_sram_in = sram_act_dout;
                else spad28_a_data_sram_in = 0;
            end
            default:spad28_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[27] == load_a) & !addr_sram_act_re_end_29) begin
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
            // spad29_a_data_sram_in = 0; 
            spad30_a_data_sram_in = 0; 
            spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;
            case (padding_crl_mem[27])
            0:begin
                if (~is_odd_mem[27]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad29_a_data_sram_in = sram_act_dout;
                    else spad29_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad29_a_data_sram_in = sram_act_dout;
                    else spad29_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad29_a_data_sram_in = sram_act_dout;
                else spad29_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad29_a_data_sram_in = sram_act_dout;
                else spad29_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad29_a_data_sram_in = sram_act_dout;
                else spad29_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad29_a_data_sram_in = sram_act_dout;
                else spad29_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad29_a_data_sram_in = sram_act_dout;
                else spad29_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad29_a_data_sram_in = sram_act_dout;
                else spad29_a_data_sram_in = 0;
            end
            default:spad29_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[28] == load_a) & !addr_sram_act_re_end_30) begin
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
            // spad30_a_data_sram_in = 0; 
            spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;
            case (padding_crl_mem[28])
            0:begin
                if (~is_odd_mem[28]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad30_a_data_sram_in = sram_act_dout;
                    else spad30_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad30_a_data_sram_in = sram_act_dout;
                    else spad30_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad30_a_data_sram_in = sram_act_dout;
                else spad30_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad30_a_data_sram_in = sram_act_dout;
                else spad30_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad30_a_data_sram_in = sram_act_dout;
                else spad30_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad30_a_data_sram_in = sram_act_dout;
                else spad30_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad30_a_data_sram_in = sram_act_dout;
                else spad30_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad30_a_data_sram_in = sram_act_dout;
                else spad30_a_data_sram_in = 0;
            end
            default:spad30_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[29] == load_a) & !addr_sram_act_re_end_31) begin
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
            // spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;
            case (padding_crl_mem[29])
            0:begin
                if (~is_odd_mem[29]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad31_a_data_sram_in = sram_act_dout;
                    else spad31_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad31_a_data_sram_in = sram_act_dout;
                    else spad31_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad31_a_data_sram_in = sram_act_dout;
                else spad31_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad31_a_data_sram_in = sram_act_dout;
                else spad31_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad31_a_data_sram_in = sram_act_dout;
                else spad31_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad31_a_data_sram_in = sram_act_dout;
                else spad31_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad31_a_data_sram_in = sram_act_dout;
                else spad31_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad31_a_data_sram_in = sram_act_dout;
                else spad31_a_data_sram_in = 0;
            end
            default:spad31_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[30] == load_a) & !addr_sram_act_re_end_32) begin
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
            // spad32_a_data_sram_in = 0;
            case (padding_crl_mem[30])
            0:begin
                if (~is_odd_mem[30]) begin
                    if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad32_a_data_sram_in = sram_act_dout;
                    else spad32_a_data_sram_in = 0;
                end
                else begin
                    if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad32_a_data_sram_in = sram_act_dout;
                    else spad32_a_data_sram_in = 0;
                end
            end
            1:begin
                if ((cnt_re_sram == 4) | (cnt_re_sram == 6)) spad32_a_data_sram_in = sram_act_dout;
                else spad32_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram == 3) | (cnt_re_sram == 5) | (cnt_re_sram == 7)) spad32_a_data_sram_in = sram_act_dout;
                else spad32_a_data_sram_in = 0;
            end
            3:begin
                if ((cnt_re_sram == 2) | (cnt_re_sram == 4) | (cnt_re_sram == 6)) spad32_a_data_sram_in = sram_act_dout;
                else spad32_a_data_sram_in = 0;
            end
            4:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) | (cnt_re_sram == 5)) spad32_a_data_sram_in = sram_act_dout;
                else spad32_a_data_sram_in = 0;
            end
            5:begin
                if ((cnt_re_sram == 0) | (cnt_re_sram == 2) | (cnt_re_sram == 4)) spad32_a_data_sram_in = sram_act_dout;
                else spad32_a_data_sram_in = 0;
            end
            6:begin
                if ((cnt_re_sram == 1) | (cnt_re_sram == 3) ) spad32_a_data_sram_in = sram_act_dout;
                else spad32_a_data_sram_in = 0;
            end
            default:spad32_a_data_sram_in = 0;
            endcase
        end
        else begin
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
        end


    end
    else if ((decoder_top_state == cnn11) | (decoder_top_state == cnn12)|decoder_top_state_8) begin
        if ((conv_state == load_a) & !addr_sram_act_re_end_1) begin
            // spad1_a_data_sram_in = 0;
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
            case (padding_crl)
            0:begin
                    if (cnt_re_sram < CNN_KS) spad1_a_data_sram_in = {{(INPUT_DW - SRAM8_DW){sram_act_dout[SRAM8_DW-1]}} , sram_act_dout};
                    else spad1_a_data_sram_in = 0;                    
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad1_a_data_sram_in = {{(INPUT_DW - SRAM8_DW){sram_act_dout[SRAM8_DW-1]}} , sram_act_dout};
                else spad1_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad1_a_data_sram_in = {{(INPUT_DW - SRAM8_DW){sram_act_dout[SRAM8_DW-1]}} , sram_act_dout};
                else spad1_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad1_a_data_sram_in = {{(INPUT_DW - SRAM8_DW){sram_act_dout[SRAM8_DW-1]}} , sram_act_dout};
                else spad1_a_data_sram_in = 0;
            end
            4:begin
                if  (cnt_re_sram < CNN_KS-2) spad1_a_data_sram_in = {{(INPUT_DW - SRAM8_DW){sram_act_dout[SRAM8_DW-1]}} , sram_act_dout};
                else spad1_a_data_sram_in = 0;
            end
            default:spad1_a_data_sram_in = 0;
            endcase
        end 
        else if ((conv_state_mem[0] == load_a) & !addr_sram_act_re_end_2) begin
            spad1_a_data_sram_in = 0;
            // spad2_a_data_sram_in = 0;        
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
            case (padding_crl_mem[0])
            0:begin
                if (cnt_re_sram < CNN_KS) spad2_a_data_sram_in = sram_act_dout;
                else spad2_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad2_a_data_sram_in = sram_act_dout;
                else spad2_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad2_a_data_sram_in = sram_act_dout;
                else spad2_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad2_a_data_sram_in = sram_act_dout;
                else spad2_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad2_a_data_sram_in = sram_act_dout;
                else spad2_a_data_sram_in = 0;
            end
            default:spad2_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[1] == load_a) & !addr_sram_act_re_end_3) begin
            spad1_a_data_sram_in = 0;
            spad2_a_data_sram_in = 0;        
            // spad3_a_data_sram_in = 0; 
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
            case (padding_crl_mem[1])
            0:begin
                if (cnt_re_sram < CNN_KS) spad3_a_data_sram_in = sram_act_dout;
                else spad3_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad3_a_data_sram_in = sram_act_dout;
                else spad3_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad3_a_data_sram_in = sram_act_dout;
                else spad3_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad3_a_data_sram_in = sram_act_dout;
                else spad3_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad3_a_data_sram_in = sram_act_dout;
                else spad3_a_data_sram_in = 0;
            end
            default:spad3_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[2] == load_a) & !addr_sram_act_re_end_4) begin
            spad1_a_data_sram_in = 0;
            spad2_a_data_sram_in = 0;        
            spad3_a_data_sram_in = 0; 
            // spad4_a_data_sram_in = 0; 
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
            case (padding_crl_mem[2])
            0:begin
                if (cnt_re_sram < CNN_KS) spad4_a_data_sram_in = sram_act_dout;
                else spad4_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad4_a_data_sram_in = sram_act_dout;
                else spad4_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad4_a_data_sram_in = sram_act_dout;
                else spad4_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad4_a_data_sram_in = sram_act_dout;
                else spad4_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad4_a_data_sram_in = sram_act_dout;
                else spad4_a_data_sram_in = 0;
            end
            default:spad4_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[3] == load_a) & !addr_sram_act_re_end_5) begin
            spad1_a_data_sram_in = 0;
            spad2_a_data_sram_in = 0;        
            spad3_a_data_sram_in = 0; 
            spad4_a_data_sram_in = 0; 
            // spad5_a_data_sram_in = 0; 
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
            case (padding_crl_mem[3])
            0:begin
                if (cnt_re_sram < CNN_KS) spad5_a_data_sram_in = sram_act_dout;
                else spad5_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad5_a_data_sram_in = sram_act_dout;
                else spad5_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad5_a_data_sram_in = sram_act_dout;
                else spad5_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad5_a_data_sram_in = sram_act_dout;
                else spad5_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad5_a_data_sram_in = sram_act_dout;
                else spad5_a_data_sram_in = 0;
            end
            default:spad5_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[4] == load_a) & !addr_sram_act_re_end_6) begin
            spad1_a_data_sram_in = 0;
            spad2_a_data_sram_in = 0;        
            spad3_a_data_sram_in = 0; 
            spad4_a_data_sram_in = 0; 
            spad5_a_data_sram_in = 0; 
            // spad6_a_data_sram_in = 0; 
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
            case (padding_crl_mem[4])
            0:begin
                if (cnt_re_sram < CNN_KS) spad6_a_data_sram_in = sram_act_dout;
                else spad6_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad6_a_data_sram_in = sram_act_dout;
                else spad6_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad6_a_data_sram_in = sram_act_dout;
                else spad6_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad6_a_data_sram_in = sram_act_dout;
                else spad6_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad6_a_data_sram_in = sram_act_dout;
                else spad6_a_data_sram_in = 0;
            end
            default:spad6_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[5] == load_a) & !addr_sram_act_re_end_7) begin
            spad1_a_data_sram_in = 0;
            spad2_a_data_sram_in = 0;        
            spad3_a_data_sram_in = 0; 
            spad4_a_data_sram_in = 0; 
            spad5_a_data_sram_in = 0; 
            spad6_a_data_sram_in = 0; 
            // spad7_a_data_sram_in = 0; 
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
            case (padding_crl_mem[5])
            0:begin
                if (cnt_re_sram < CNN_KS) spad7_a_data_sram_in = sram_act_dout;
                else spad7_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad7_a_data_sram_in = sram_act_dout;
                else spad7_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad7_a_data_sram_in = sram_act_dout;
                else spad7_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad7_a_data_sram_in = sram_act_dout;
                else spad7_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad7_a_data_sram_in = sram_act_dout;
                else spad7_a_data_sram_in = 0;
            end
            default:spad7_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[6] == load_a) & !addr_sram_act_re_end_8) begin
            spad1_a_data_sram_in = 0;
            spad2_a_data_sram_in = 0;        
            spad3_a_data_sram_in = 0; 
            spad4_a_data_sram_in = 0; 
            spad5_a_data_sram_in = 0; 
            spad6_a_data_sram_in = 0; 
            spad7_a_data_sram_in = 0; 
            // spad8_a_data_sram_in = 0; 
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
            case (padding_crl_mem[6])
            0:begin
                if (cnt_re_sram < CNN_KS) spad8_a_data_sram_in = sram_act_dout;
                else spad8_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad8_a_data_sram_in = sram_act_dout;
                else spad8_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad8_a_data_sram_in = sram_act_dout;
                else spad8_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad8_a_data_sram_in = sram_act_dout;
                else spad8_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad8_a_data_sram_in = sram_act_dout;
                else spad8_a_data_sram_in = 0;
            end
            default:spad8_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[7] == load_a) & !addr_sram_act_re_end_9) begin
            spad1_a_data_sram_in = 0;
            spad2_a_data_sram_in = 0;        
            spad3_a_data_sram_in = 0; 
            spad4_a_data_sram_in = 0; 
            spad5_a_data_sram_in = 0; 
            spad6_a_data_sram_in = 0; 
            spad7_a_data_sram_in = 0; 
            spad8_a_data_sram_in = 0; 
            // spad9_a_data_sram_in = 0; 
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
            case (padding_crl_mem[7])
            0:begin
                if (cnt_re_sram < CNN_KS) spad9_a_data_sram_in = sram_act_dout;
                else spad9_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad9_a_data_sram_in = sram_act_dout;
                else spad9_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad9_a_data_sram_in = sram_act_dout;
                else spad9_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad9_a_data_sram_in = sram_act_dout;
                else spad9_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad9_a_data_sram_in = sram_act_dout;
                else spad9_a_data_sram_in = 0;
            end
            default:spad9_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[8] == load_a) & !addr_sram_act_re_end_10) begin
            spad1_a_data_sram_in = 0;
            spad2_a_data_sram_in = 0;        
            spad3_a_data_sram_in = 0; 
            spad4_a_data_sram_in = 0; 
            spad5_a_data_sram_in = 0; 
            spad6_a_data_sram_in = 0; 
            spad7_a_data_sram_in = 0; 
            spad8_a_data_sram_in = 0; 
            spad9_a_data_sram_in = 0; 
            // spad10_a_data_sram_in = 0; 
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
            case (padding_crl_mem[8])
            0:begin
                if (cnt_re_sram < CNN_KS) spad10_a_data_sram_in = sram_act_dout;
                else spad10_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad10_a_data_sram_in = sram_act_dout;
                else spad10_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad10_a_data_sram_in = sram_act_dout;
                else spad10_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad10_a_data_sram_in = sram_act_dout;
                else spad10_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad10_a_data_sram_in = sram_act_dout;
                else spad10_a_data_sram_in = 0;
            end
            default:spad10_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[9] == load_a) & !addr_sram_act_re_end_11) begin
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
            // spad11_a_data_sram_in = 0; 
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
            case (padding_crl_mem[9])
            0:begin
                if (cnt_re_sram < CNN_KS) spad11_a_data_sram_in = sram_act_dout;
                else spad11_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad11_a_data_sram_in = sram_act_dout;
                else spad11_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad11_a_data_sram_in = sram_act_dout;
                else spad11_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad11_a_data_sram_in = sram_act_dout;
                else spad11_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad11_a_data_sram_in = sram_act_dout;
                else spad11_a_data_sram_in = 0;
            end
            default:spad11_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[10] == load_a) & !addr_sram_act_re_end_12) begin
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
            // spad12_a_data_sram_in = 0; 
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
            case (padding_crl_mem[10])
            0:begin
                if (cnt_re_sram < CNN_KS) spad12_a_data_sram_in = sram_act_dout;
                else spad12_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad12_a_data_sram_in = sram_act_dout;
                else spad12_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad12_a_data_sram_in = sram_act_dout;
                else spad12_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad12_a_data_sram_in = sram_act_dout;
                else spad12_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad12_a_data_sram_in = sram_act_dout;
                else spad12_a_data_sram_in = 0;
            end
            default:spad12_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[11] == load_a) & !addr_sram_act_re_end_13) begin
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
            // spad13_a_data_sram_in = 0; 
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
            case (padding_crl_mem[11])
            0:begin
                if (cnt_re_sram < CNN_KS) spad13_a_data_sram_in = sram_act_dout;
                else spad13_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad13_a_data_sram_in = sram_act_dout;
                else spad13_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad13_a_data_sram_in = sram_act_dout;
                else spad13_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad13_a_data_sram_in = sram_act_dout;
                else spad13_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad13_a_data_sram_in = sram_act_dout;
                else spad13_a_data_sram_in = 0;
            end
            default:spad13_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[12] == load_a) & !addr_sram_act_re_end_14) begin
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
            // spad14_a_data_sram_in = 0; 
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
            case (padding_crl_mem[12])
            0:begin
                if (cnt_re_sram < CNN_KS) spad14_a_data_sram_in = sram_act_dout;
                else spad14_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad14_a_data_sram_in = sram_act_dout;
                else spad14_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad14_a_data_sram_in = sram_act_dout;
                else spad14_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad14_a_data_sram_in = sram_act_dout;
                else spad14_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad14_a_data_sram_in = sram_act_dout;
                else spad14_a_data_sram_in = 0;
            end
            default:spad14_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[13] == load_a) & !addr_sram_act_re_end_15) begin
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
            // spad15_a_data_sram_in = 0; 
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
            case (padding_crl_mem[13])
            0:begin
                if (cnt_re_sram < CNN_KS) spad15_a_data_sram_in = sram_act_dout;
                else spad15_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad15_a_data_sram_in = sram_act_dout;
                else spad15_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad15_a_data_sram_in = sram_act_dout;
                else spad15_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad15_a_data_sram_in = sram_act_dout;
                else spad15_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad15_a_data_sram_in = sram_act_dout;
                else spad15_a_data_sram_in = 0;
            end
            default:spad15_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[14] == load_a) & !addr_sram_act_re_end_16) begin
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
            // spad16_a_data_sram_in = 0; 
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
            case (padding_crl_mem[14])
            0:begin
                if (cnt_re_sram < CNN_KS) spad16_a_data_sram_in = sram_act_dout;
                else spad16_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad16_a_data_sram_in = sram_act_dout;
                else spad16_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad16_a_data_sram_in = sram_act_dout;
                else spad16_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad16_a_data_sram_in = sram_act_dout;
                else spad16_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad16_a_data_sram_in = sram_act_dout;
                else spad16_a_data_sram_in = 0;
            end
            default:spad16_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[15] == load_a) & !addr_sram_act_re_end_17) begin
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
            // spad17_a_data_sram_in = 0; 
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
            case (padding_crl_mem[15])
            0:begin
                if (cnt_re_sram < CNN_KS) spad17_a_data_sram_in = sram_act_dout;
                else spad17_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad17_a_data_sram_in = sram_act_dout;
                else spad17_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad17_a_data_sram_in = sram_act_dout;
                else spad17_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad17_a_data_sram_in = sram_act_dout;
                else spad17_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad17_a_data_sram_in = sram_act_dout;
                else spad17_a_data_sram_in = 0;
            end
            default:spad17_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[16] == load_a) & !addr_sram_act_re_end_18) begin
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
            // spad18_a_data_sram_in = 0; 
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
            case (padding_crl_mem[16])
            0:begin
                if (cnt_re_sram < CNN_KS) spad18_a_data_sram_in = sram_act_dout;
                else spad18_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad18_a_data_sram_in = sram_act_dout;
                else spad18_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad18_a_data_sram_in = sram_act_dout;
                else spad18_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad18_a_data_sram_in = sram_act_dout;
                else spad18_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad18_a_data_sram_in = sram_act_dout;
                else spad18_a_data_sram_in = 0;
            end
            default:spad18_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[17] == load_a) & !addr_sram_act_re_end_19) begin
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
            // spad19_a_data_sram_in = 0; 
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
            case (padding_crl_mem[17])
            0:begin
                if (cnt_re_sram < CNN_KS) spad19_a_data_sram_in = sram_act_dout;
                else spad19_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad19_a_data_sram_in = sram_act_dout;
                else spad19_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad19_a_data_sram_in = sram_act_dout;
                else spad19_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad19_a_data_sram_in = sram_act_dout;
                else spad19_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad19_a_data_sram_in = sram_act_dout;
                else spad19_a_data_sram_in = 0;
            end
            default:spad19_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[18] == load_a) & !addr_sram_act_re_end_20) begin
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
            // spad20_a_data_sram_in = 0; 
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
            case (padding_crl_mem[18])
            0:begin
                if (cnt_re_sram < CNN_KS) spad20_a_data_sram_in = sram_act_dout;
                else spad20_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad20_a_data_sram_in = sram_act_dout;
                else spad20_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad20_a_data_sram_in = sram_act_dout;
                else spad20_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad20_a_data_sram_in = sram_act_dout;
                else spad20_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad20_a_data_sram_in = sram_act_dout;
                else spad20_a_data_sram_in = 0;
            end
            default:spad20_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[19] == load_a) & !addr_sram_act_re_end_21) begin
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
            // spad21_a_data_sram_in = 0; 
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
            case (padding_crl_mem[19])
            0:begin
                if (cnt_re_sram < CNN_KS) spad21_a_data_sram_in = sram_act_dout;
                else spad21_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad21_a_data_sram_in = sram_act_dout;
                else spad21_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad21_a_data_sram_in = sram_act_dout;
                else spad21_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad21_a_data_sram_in = sram_act_dout;
                else spad21_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad21_a_data_sram_in = sram_act_dout;
                else spad21_a_data_sram_in = 0;
            end
            default:spad21_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[20] == load_a) & !addr_sram_act_re_end_22) begin
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
            // spad22_a_data_sram_in = 0; 
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
            case (padding_crl_mem[20])
            0:begin
                if (cnt_re_sram < CNN_KS) spad22_a_data_sram_in = sram_act_dout;
                else spad22_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad22_a_data_sram_in = sram_act_dout;
                else spad22_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad22_a_data_sram_in = sram_act_dout;
                else spad22_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad22_a_data_sram_in = sram_act_dout;
                else spad22_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad22_a_data_sram_in = sram_act_dout;
                else spad22_a_data_sram_in = 0;
            end
            default:spad22_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[21] == load_a) & !addr_sram_act_re_end_23) begin
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
            // spad23_a_data_sram_in = 0; 
            spad24_a_data_sram_in = 0;
            spad25_a_data_sram_in = 0;
            spad26_a_data_sram_in = 0; 
            spad27_a_data_sram_in = 0; 
            spad28_a_data_sram_in = 0; 
            spad29_a_data_sram_in = 0; 
            spad30_a_data_sram_in = 0; 
            spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;    
            case (padding_crl_mem[21])
            0:begin
                if (cnt_re_sram < CNN_KS) spad23_a_data_sram_in = sram_act_dout;
                else spad23_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad23_a_data_sram_in = sram_act_dout;
                else spad23_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad23_a_data_sram_in = sram_act_dout;
                else spad23_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad23_a_data_sram_in = sram_act_dout;
                else spad23_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad23_a_data_sram_in = sram_act_dout;
                else spad23_a_data_sram_in = 0;
            end
            default:spad23_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[22] == load_a) & !addr_sram_act_re_end_24) begin
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
            // spad24_a_data_sram_in = 0;
            spad25_a_data_sram_in = 0;
            spad26_a_data_sram_in = 0; 
            spad27_a_data_sram_in = 0; 
            spad28_a_data_sram_in = 0; 
            spad29_a_data_sram_in = 0; 
            spad30_a_data_sram_in = 0; 
            spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;    
            case (padding_crl_mem[22])
            0:begin
                if (cnt_re_sram < CNN_KS) spad24_a_data_sram_in = sram_act_dout;
                else spad24_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad24_a_data_sram_in = sram_act_dout;
                else spad24_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad24_a_data_sram_in = sram_act_dout;
                else spad24_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad24_a_data_sram_in = sram_act_dout;
                else spad24_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad24_a_data_sram_in = sram_act_dout;
                else spad24_a_data_sram_in = 0;
            end
            default:spad24_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[23] == load_a) & !addr_sram_act_re_end_25) begin
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
            // spad25_a_data_sram_in = 0;
            spad26_a_data_sram_in = 0; 
            spad27_a_data_sram_in = 0; 
            spad28_a_data_sram_in = 0; 
            spad29_a_data_sram_in = 0; 
            spad30_a_data_sram_in = 0; 
            spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;    
            case (padding_crl_mem[23])
            0:begin
                if (cnt_re_sram < CNN_KS) spad25_a_data_sram_in = sram_act_dout;
                else spad25_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad25_a_data_sram_in = sram_act_dout;
                else spad25_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad25_a_data_sram_in = sram_act_dout;
                else spad25_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad25_a_data_sram_in = sram_act_dout;
                else spad25_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad25_a_data_sram_in = sram_act_dout;
                else spad25_a_data_sram_in = 0;
            end
            default:spad25_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[24] == load_a) & !addr_sram_act_re_end_26) begin
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
            // spad26_a_data_sram_in = 0; 
            spad27_a_data_sram_in = 0; 
            spad28_a_data_sram_in = 0; 
            spad29_a_data_sram_in = 0; 
            spad30_a_data_sram_in = 0; 
            spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;    
            case (padding_crl_mem[24])
            0:begin
                if (cnt_re_sram < CNN_KS) spad26_a_data_sram_in = sram_act_dout;
                else spad26_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad26_a_data_sram_in = sram_act_dout;
                else spad26_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad26_a_data_sram_in = sram_act_dout;
                else spad26_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad26_a_data_sram_in = sram_act_dout;
                else spad26_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad26_a_data_sram_in = sram_act_dout;
                else spad26_a_data_sram_in = 0;
            end
            default:spad26_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[25] == load_a) & !addr_sram_act_re_end_27) begin
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
            // spad27_a_data_sram_in = 0; 
            spad28_a_data_sram_in = 0; 
            spad29_a_data_sram_in = 0; 
            spad30_a_data_sram_in = 0; 
            spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;    
            case (padding_crl_mem[25])
            0:begin
                if (cnt_re_sram < CNN_KS) spad27_a_data_sram_in = sram_act_dout;
                else spad27_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad27_a_data_sram_in = sram_act_dout;
                else spad27_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad27_a_data_sram_in = sram_act_dout;
                else spad27_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad27_a_data_sram_in = sram_act_dout;
                else spad27_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad27_a_data_sram_in = sram_act_dout;
                else spad27_a_data_sram_in = 0;
            end
            default:spad27_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[26] == load_a) & !addr_sram_act_re_end_28) begin
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
            // spad28_a_data_sram_in = 0; 
            spad29_a_data_sram_in = 0; 
            spad30_a_data_sram_in = 0; 
            spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;    
            case (padding_crl_mem[26])
            0:begin
                if (cnt_re_sram < CNN_KS) spad28_a_data_sram_in = sram_act_dout;
                else spad28_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad28_a_data_sram_in = sram_act_dout;
                else spad28_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad28_a_data_sram_in = sram_act_dout;
                else spad28_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad28_a_data_sram_in = sram_act_dout;
                else spad28_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad28_a_data_sram_in = sram_act_dout;
                else spad28_a_data_sram_in = 0;
            end
            default:spad28_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[27] == load_a) & !addr_sram_act_re_end_29) begin
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
            // spad29_a_data_sram_in = 0; 
            spad30_a_data_sram_in = 0; 
            spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;    
            case (padding_crl_mem[27])
            0:begin
                if (cnt_re_sram < CNN_KS) spad29_a_data_sram_in = sram_act_dout;
                else spad29_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad29_a_data_sram_in = sram_act_dout;
                else spad29_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad29_a_data_sram_in = sram_act_dout;
                else spad29_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad29_a_data_sram_in = sram_act_dout;
                else spad29_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad29_a_data_sram_in = sram_act_dout;
                else spad29_a_data_sram_in = 0;
            end
            default:spad29_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[28] == load_a) & !addr_sram_act_re_end_30) begin
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
            // spad30_a_data_sram_in = 0; 
            spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;    
            case (padding_crl_mem[28])
            0:begin
                if (cnt_re_sram < CNN_KS) spad30_a_data_sram_in = sram_act_dout;
                else spad30_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad30_a_data_sram_in = sram_act_dout;
                else spad30_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad30_a_data_sram_in = sram_act_dout;
                else spad30_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad30_a_data_sram_in = sram_act_dout;
                else spad30_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad30_a_data_sram_in = sram_act_dout;
                else spad30_a_data_sram_in = 0;
            end
            default:spad30_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[29] == load_a) & !addr_sram_act_re_end_31) begin
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
            // spad31_a_data_sram_in = 0; 
            spad32_a_data_sram_in = 0;    
            case (padding_crl_mem[29])
            0:begin
                if (cnt_re_sram < CNN_KS) spad31_a_data_sram_in = sram_act_dout;
                else spad31_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad31_a_data_sram_in = sram_act_dout;
                else spad31_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad31_a_data_sram_in = sram_act_dout;
                else spad31_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad31_a_data_sram_in = sram_act_dout;
                else spad31_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad31_a_data_sram_in = sram_act_dout;
                else spad31_a_data_sram_in = 0;
            end
            default:spad31_a_data_sram_in = 0;
            endcase
        end
        else if ((conv_state_mem[30] == load_a) & !addr_sram_act_re_end_32) begin
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
            // spad32_a_data_sram_in = 0;    
            case (padding_crl_mem[30])
            0:begin
                if (cnt_re_sram < CNN_KS) spad32_a_data_sram_in = sram_act_dout;
                else spad32_a_data_sram_in = 0;
            end
            1:begin
                if ((cnt_re_sram > 1) & (cnt_re_sram < CNN_KS)) spad32_a_data_sram_in = sram_act_dout;
                else spad32_a_data_sram_in = 0;
            end
            2:begin
                if ((cnt_re_sram > 0) & (cnt_re_sram < CNN_KS)) spad32_a_data_sram_in = sram_act_dout;
                else spad32_a_data_sram_in = 0;
            end
            3:begin
                if (cnt_re_sram < CNN_KS-1) spad32_a_data_sram_in = sram_act_dout;
                else spad32_a_data_sram_in = 0;
            end
            4:begin
                if (cnt_re_sram < CNN_KS-2) spad32_a_data_sram_in = sram_act_dout;
                else spad32_a_data_sram_in = 0;
            end
            default:spad32_a_data_sram_in = 0;
            endcase
        end
        else begin
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
        end


    end

    else begin
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
    end
end

// wire layer_out_vld_half;
// assign layer_out_vld_half = !wclk & layer_out_vld;
reg layer_out_vld_d;
reg [DATA_DW-1: 0] decoder_out_reg; // dcnn1, cnn11, cnn12
reg [2*DATA_DW-1: 0] decoder_out_cat_reg; // cnn21, cnn22
wire layer_out_vld_half;
assign layer_out_vld_half = wclk & layer_out_vld_d;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        decoder_out_reg <= 0;
        decoder_out_cat_reg <= 0;
    end
    else begin
        decoder_out_reg <= decoder_out;
        decoder_out_cat_reg <= decoder_out_cat;
    end
end

always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) layer_out_vld_d <= 0;
    else layer_out_vld_d <= layer_out_vld;
end 

reg is_pre;
always @(negedge sclk or negedge rst_n) begin
    if (!rst_n) begin
        addr_sram_act_we <= ADDR_DCNN1_SRAM_ACT_INIT; //change
        // addr_sram_act_we <= ADDR_CNN21_SRAM_ACT_INIT;//change
        addr_sram_act_we_end <= 0;
        cnt_we_sram <= 0;
        is_pre <= 1;
    end
    else begin
        if (decoder_top_state_32) begin
            if (layer_out_vld_half) begin
                if (!addr_sram_act_we_end) begin
                        addr_sram_act_we <= addr_sram_act_we + 1;
                        addr_sram_act_we_end <= 1;                 
                end
                else begin
                    addr_sram_act_we <= addr_sram_act_we;
                    addr_sram_act_we_end <= addr_sram_act_we_end;               
                end
            end
            else if (layer_done) begin
                if (decoder_top_state == dcnn1) begin
                    addr_sram_act_we <= ADDR_CNN11_SRAM_ACT_INIT_0;
                end
                else if (decoder_top_state == cnn11) begin
                    addr_sram_act_we <= ADDR_CNN12_SRAM_ACT_INIT;
                end
                else if (decoder_top_state == cnn12) begin
                    addr_sram_act_we <= ADDR_DCNN2_SRAM_ACT_INIT;
                end
                
                addr_sram_act_we_end <= 0;                
            end
            else begin
                addr_sram_act_we_end <= 0;
                addr_sram_act_we <= addr_sram_act_we;                 
            end

        end
        else if (decoder_top_state_16) begin
            if (layer_out_vld_half) begin
                if (!addr_sram_act_we_end) begin
                    if (cnt_we_sram == 1) begin
                        addr_sram_act_we <= addr_sram_act_we + 1 - DCNN2_CHOUT/2*DCNN2_LENGTH_OUT;
                        addr_sram_act_we_end <= 1;
                        cnt_we_sram <= 0;
                    end
                    else begin
                        addr_sram_act_we <= addr_sram_act_we + DCNN2_CHOUT/2*DCNN2_LENGTH_OUT;
                        addr_sram_act_we_end <= 0;
                        cnt_we_sram <= cnt_we_sram + 1;                    
                    end
                end
                else begin
                    addr_sram_act_we <= addr_sram_act_we;
                    addr_sram_act_we_end <= addr_sram_act_we_end;
                    cnt_we_sram <= cnt_we_sram;                      
                end                
            end
            else if (layer_done) begin
                addr_sram_act_we <= ADDR_CNN21_SRAM_ACT_INIT;
                addr_sram_act_we_end <= 0;                
            end
            else begin
                addr_sram_act_we <= addr_sram_act_we;
                addr_sram_act_we_end <= 0;                  
            end
        end
        else if (decoder_top_state == cnn21) begin
            if (layer_out_vld_half) begin
                if (!addr_sram_act_we_end) begin
                    if (cnt_we_sram == 1) begin
                        addr_sram_act_we <= (is_pre)? addr_sram_act_we + CNN21_CHOUT/4*CNN21_LENGTH_OUT : addr_sram_act_we + 1 -3*CNN21_CHOUT/4*CNN21_LENGTH_OUT;
                        addr_sram_act_we_end <= 1;
                        cnt_we_sram <= 0;
                        is_pre <= ~is_pre;
                    end
                    else begin
                        addr_sram_act_we <= addr_sram_act_we + CNN21_CHOUT/4*CNN21_LENGTH_OUT;
                        addr_sram_act_we_end <= 0;
                        cnt_we_sram <= cnt_we_sram + 1; 
                        is_pre <= is_pre;                   
                    end
                end
                else begin
                    addr_sram_act_we <= addr_sram_act_we;
                    addr_sram_act_we_end <= addr_sram_act_we_end;
                    cnt_we_sram <= cnt_we_sram; 
                    is_pre <= is_pre;    
                
                end                
            end
            // else if (layer_done) begin
            //     addr_sram_act_we <= ADDR_DCNN1_SRAM_ACT_INIT; //end for the next round
            //     addr_sram_act_we_end <= 0; 
            //     is_pre <= 1;            
            // end
            else begin
                addr_sram_act_we <= addr_sram_act_we;
                addr_sram_act_we_end <= 0;
                is_pre <= is_pre; 
                             
            end
        end
        else if (decoder_top_state == idle)begin
            addr_sram_act_we <= ADDR_DCNN1_SRAM_ACT_INIT; //change
            addr_sram_act_we_end <= 0; 
            is_pre <= 1;
            cnt_we_sram <= 0;                
        end
        else begin
            addr_sram_act_we <= addr_sram_act_we;
            addr_sram_act_we_end <= addr_sram_act_we_end;
            is_pre <= is_pre;  
            cnt_we_sram <= cnt_we_sram;          
        end

    end
end
// assign sram_act_din = (layer_out_vld & !addr_sram_act_we_end)?decoder_out:0;
always @(*) begin
    if (decoder_top_state_32) begin
        sram_act_din = (sram_act_we)?decoder_out_reg : 0;
    end
    else if (decoder_top_state == dcnn2|decoder_top_state_8) begin
        sram_act_din = (sram_act_we)?((cnt_we_sram)? decoder_out_cat_reg[2*DATA_DW-1:DATA_DW] : decoder_out_cat_reg[DATA_DW-1:0]):0; 
    end
    else sram_act_din = 0;
end
assign sram_act_en = layer_out_vld_half | (conv_state == load_a) | (conv_state_mem[0] == load_a) |
                    (conv_state_mem[1] == load_a) | (conv_state_mem[2] == load_a) | (conv_state_mem[3] == load_a) |
                    (conv_state_mem[4] == load_a) | (conv_state_mem[5] == load_a) | (conv_state_mem[6] == load_a) |
                    (conv_state_mem[7] == load_a) | (conv_state_mem[8] == load_a) | (conv_state_mem[9] == load_a) |
                    (conv_state_mem[10] == load_a) | (conv_state_mem[11] == load_a) | (conv_state_mem[12] == load_a) |
                    (conv_state_mem[13] == load_a) | (conv_state_mem[14] == load_a) | (conv_state_mem[15] == load_a) |
                    (conv_state_mem[16] == load_a) | (conv_state_mem[17] == load_a) | (conv_state_mem[18] == load_a) |
                    (conv_state_mem[19] == load_a) | (conv_state_mem[20] == load_a) | (conv_state_mem[21] == load_a) |
                    (conv_state_mem[22] == load_a) | (conv_state_mem[23] == load_a) | (conv_state_mem[24] == load_a) |
                    (conv_state_mem[25] == load_a) | (conv_state_mem[26] == load_a) | (conv_state_mem[27] == load_a) |
                    (conv_state_mem[28] == load_a) | (conv_state_mem[29] == load_a) | (conv_state_mem[30] == load_a);
assign sram_act_we = (layer_out_vld_half & !addr_sram_act_we_end)? 1:0;
assign addr_sram_act = sram_act_we? addr_sram_act_we:addr_sram_act_re;


    always@(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            spad_w_addr_re <= 0 ;
            spad_a_addr_re <= 0;
        end
        else begin
            if (conv_state == mac) begin
                spad_w_addr_re <= (decoder_top_state_dcnn)?((spad_w_addr_re == DCNN_KS-1)? spad_w_addr_re: spad_w_addr_re + 1):((spad_w_addr_re == CNN_KS-1)? spad_w_addr_re: spad_w_addr_re + 1);
                spad_a_addr_re <= (decoder_top_state_dcnn)?((spad_a_addr_re == DCNN_KS-1)? spad_a_addr_re:spad_a_addr_re + 1):((spad_a_addr_re == CNN_KS-1)? spad_a_addr_re: spad_a_addr_re + 1);
            end
            else begin
                spad_w_addr_re <= 0;
                spad_a_addr_re <= 0;
            end
        end
    end

//mult_int8_crl:000:idle/reset, 001:gatess_mac, 011: transfer, 111:hold,
reg [2:0] mult_int8_crl_n;
reg [2:0] mult_int8_crl;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n)
        mult_int8_crl <= idle;
    else
        mult_int8_crl <= mult_int8_crl_n;
end
always @(*) begin
    case (mult_int8_crl)
    3'b000: begin
        if (conv_state_next == mac)  
            mult_int8_crl_n = 3'b001;
        else
            mult_int8_crl_n = 3'b000;        
    end
    3'b001:begin
        if(decoder_top_state_dcnn) begin
            if (cnt_ks  == DCNN_KS) 
                    mult_int8_crl_n = 3'b011; // transfer
            else
                mult_int8_crl_n = 3'b001; 
        end  
        else begin
            if (cnt_ks  == CNN_KS) 
                    mult_int8_crl_n = 3'b011; 
            else
                mult_int8_crl_n = 3'b001;             
        end     
    end
    3'b011: begin
        mult_int8_crl_n = 3'b000;
    end
    default:mult_int8_crl_n = 3'b000;  
    endcase 
end

reg [(PE_NUM-1)*3-1:0] mult_int8_crl_2_32;
always @(*) begin
    if (decoder_top_state_32)  mult_int8_crl_all = {mult_int8_crl_2_32, mult_int8_crl};
    else if (decoder_top_state_16) mult_int8_crl_all =  {mult_int8_crl_2_32[(PE_NUM/2-1)*3-1:0],mult_int8_crl,mult_int8_crl_2_32[(PE_NUM/2-1)*3-1:0],mult_int8_crl};
    else if (decoder_top_state_8) mult_int8_crl_all = {mult_int8_crl_2_32[(PE_NUM/4-1)*3-1:0],mult_int8_crl,mult_int8_crl_2_32[(PE_NUM/4-1)*3-1:0],mult_int8_crl,mult_int8_crl_2_32[(PE_NUM/4-1)*3-1:0],mult_int8_crl,mult_int8_crl_2_32[(PE_NUM/4-1)*3-1:0],mult_int8_crl};
    else  mult_int8_crl_all = 0;
end

// assign mult_int8_crl_all = {mult_int8_crl_2_32, mult_int8_crl};
integer d;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin 
        mult_int8_crl_2_32 <= 0;
    end
    else begin
        mult_int8_crl_2_32[2:0] <= mult_int8_crl;
        for (d = 1; d < PE_NUM-1; d = d+1) begin
            mult_int8_crl_2_32[3*(d+1)-1-:3] <= mult_int8_crl_2_32[3*d-1-:3];
        end
    end
end

// wire [N-1:0] hhh;
// wire [3-1:0] ggg;
// assign hhh = conv_state_mem[PE_NUM-2];
// assign ggg = mult_int8_crl_2_32[(PE_NUM-1)*3-1-:3];


assign dcnn1_temp_value_for_1 = (decoder_top_state == dcnn1) ?act_sr1_1:0;
assign dcnn1_transfer_temp_value_en = (decoder_top_state == dcnn1) ? (((cnt_bt_32 == 1) & (conv_state  == pe_rst) & (mult_int8_crl ==3'b011 ))? 1:0):0;


assign dcnn1_temp_rst =(decoder_top_state == dcnn1) ? ( (cnt_lo == DCNN1_LENGTH_OUT -1) & (conv_state  == pe_rst) & (cnt_bt  == 1)):0;
always@(posedge wclk or negedge rst_n)
begin
    if (!rst_n) dcnn1_temp_value_vld <= 0;
    else
    begin
        if (decoder_top_state == dcnn1) begin
            if  ((cnt_bt_32 == 0)&(conv_state_mem[PE_NUM-2] == pe_rst) & (mult_int8_crl_2_32[(PE_NUM-1)*3-1-:3] == 3'b011 )) dcnn1_temp_value_vld <= 1;
            else dcnn1_temp_value_vld <= 0;
        end
        else begin
            dcnn1_temp_value_vld <= 0;
        end

    end
end

always@(posedge wclk or negedge rst_n)
begin
    if (!rst_n) mult_a_crl <= 0;
    else
    begin
        if  (decoder_top_state == dcnn1) begin
            if  ((cnt_bt_32 == 1)&(conv_state_mem[PE_NUM-2] == pe_rst) & (mult_int8_crl_2_32[(PE_NUM-1)*3-1-:3] == 3'b011 )) mult_a_crl <= 2'b10;
            else mult_a_crl <= 0;
        end
        else if (decoder_top_state_16) begin
            if  ((conv_state_mem[PE_NUM/2-2] == pe_rst) & (mult_int8_crl_2_32[(PE_NUM/2-1)*3-1-:3] == 3'b011 )) mult_a_crl <= 2'b10;
            else mult_a_crl <= 0;               
        end
        else begin
            mult_a_crl <= add_a_crl;
        end

    end
end
// always@(posedge wclk or negedge rst_n)
// begin
//     if (!rst_n) begin
//         mult_a_crl_d <= 0;
//     end 
//     else
//     begin
//         mult_a_crl_d <= mult_a_crl_temp;
//     end
// end

// always@(*) begin
//     if (decoder_top_state_32 |decoder_top_state_16 ) mult_a_crl = mult_a_crl_temp;
//     else if (decoder_top_state_8) mult_a_crl = mult_a_crl_temp | mult_a_crl_d;
//     else;
// end

always@(*) begin
    if (decoder_top_state_32) mult_b_crl =0;
    else if (decoder_top_state_16) mult_b_crl = mult_a_crl;
    else if (decoder_top_state_8)  mult_b_crl = mult_a_crl;
    else mult_b_crl = 0;
end

// 00 idle, 10:add_b
reg [1:0] add_a_crl_d;
reg [1:0] add_a_crl_temp;
always@(posedge wclk or negedge rst_n)
begin
    if (!rst_n) begin
        add_a_crl_temp <= 0;
    end 
    else
    begin
        if ((decoder_top_state == cnn11)|(decoder_top_state == cnn12)) begin
            if ((conv_state_mem[PE_NUM-2] == pe_rst) & (mult_int8_crl_2_32[(PE_NUM-1)*3-1-:3] == 3'b011 )) begin
                add_a_crl_temp <= 2'b10;
            end
            else begin
                add_a_crl_temp <= 0;
            end
        end
        else if ((decoder_top_state == cnn21)| (decoder_top_state == cnn22)) begin
            if ((conv_state_mem[PE_NUM/4-2] == pe_rst) & (mult_int8_crl_2_32[(PE_NUM/4-1)*3-1-:3] == 3'b011 )) begin
                add_a_crl_temp <= 2'b10;
            end
            else begin
                add_a_crl_temp <= 0;
            end            
        end
        else begin
            add_a_crl_temp <= 0;
        end
            
    end
end

always@(posedge wclk or negedge rst_n)
begin
    if (!rst_n) begin
        add_a_crl_d <= 0;
    end 
    else
    begin
        add_a_crl_d <= add_a_crl_temp;
    end
end

always@(*) begin
    if (decoder_top_state_32 ) add_a_crl = add_a_crl_temp;
    else if (decoder_top_state_8) add_a_crl = add_a_crl_temp | add_a_crl_d;
    else add_a_crl =0;
end

assign add_b_crl =(decoder_top_state_8)? add_a_crl:0;
always@(posedge wclk or negedge rst_n)
begin
    if (!rst_n) layer_out_vld <= 0;
    else
    begin


            if (mult_a_crl == 2'b10) layer_out_vld <= 1;
            else layer_out_vld <= 0;            


        // if (decoder_top_state_dcnn)  begin
        //     if (mult_a_crl == 2'b10) layer_out_vld <= 1;
        //     else layer_out_vld <= 0;
        // end
        // else if (decoder_top_state_cnn) begin
        //     if (add_a_crl == 2'b10) layer_out_vld <= 1;
        //     else layer_out_vld <= 0;            
        // end
        // else begin
        //     layer_out_vld <= 0;
        // end


    end
end

assign mult_out_round_en =  layer_out_vld ;
// assign sum_a_final_en = (mult_a_rl == 2'b10)? layer_out_vld : 0;
// assign sum_b_final_en = (decoder_top_state == cnn21|decoder_top_state == cnn22)?sum_a_final_en:0;
assign cnn22_is_first = (decoder_top_state_8)? ((add_a_crl_temp == 2'b10)? 1 : 0) : 0;

always @(*) begin
    if ((decoder_top_state == cnn22)|(decoder_top_state == cnn21)) begin
        if (cnn22_is_first) begin
            decoder_b1 = decoder_b1_par;
            decoder_b2 = decoder_b2_par;
        end
        else begin
            decoder_b1 = decoder_b3_par_d;
            decoder_b2 =  decoder_b4_par_d;       
        end
    end
    else begin
        decoder_b1 = decoder_b1_par;
        decoder_b2 = decoder_b2_par;  
    end
    
end
endmodule