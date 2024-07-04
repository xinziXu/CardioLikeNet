module LSTM_UNIT #( parameter DATA_DW = 8,
    INPUT_DW = 12,
    DATA_BQ_DW = 32,
    WU_DW = 8,
    B_DW = 32,
    SCALE_DW = 32,
    HS = 32,
    INPUT_SIZE = 32,
    SRAM1024_AW = 10,
    SRAM8192_AW = 13,
    SRAM32_DW = 32, 
    SRAM8_DW = 8,
    SRAM16_DW = 16,
    SPAD_DEPTH = 8,  
    PE_NUM = 32, 
    T = 64,
    ADDR_ENCODER_SRAM_ACT_INIT = 0,
    ADDR_LSTM10_SRAM_ACT_INIT = T * INPUT_SIZE + 2 * T * HS,
    ADDR_LSTM11_SRAM_ACT_INIT = 0) 
    (input wclk,
    input sclk,
    input rst_n,
    input [2:0] lstm_top_state,
    // input [ACTIVATION_BUF_LEN2-1:0] act_sr2, //from segment.v, XT_0, HT_0_REVERSE
    // input [ACTIVATION_BUF_LEN1-1:0] act_sr1, // from segment.v , HT_0
    // input [ACTIVATION_BUF_LEN3*DATA_DW-1:0] act_sr3,
    input [SRAM8192_AW-1:0] addr_lstm_w_init,    // from top.v, w0
    input [SRAM8192_AW-1:0] addr_lstm_u_init,    // from top.v, w0
    input [SRAM1024_AW-1:0] addr_lstm_b_init,    // from top.v, b0
    input [SRAM1024_AW-1:0] addr_lstm_scales_init,  // from top.v , SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,ct_scale,gates_scale
    input [SRAM32_DW-1 : 0] sram1_dout,   //sram1_dout: b and scales
    input [SRAM16_DW-1 : 0] sram2_dout,   //sram2_dout:  w and u
    input [SRAM16_DW-1 : 0] sram3_dout,
    input [SRAM16_DW-1 : 0] sram4_dout,
    input layer_rdy, // lstm.v


    // output signed [2*DATA_DW-1: 0] lstm_hidden_cat, //ht
    // output lstm_hidden_unit_vld,
    output reg [SRAM8192_AW-1:0] addr_sram,    // data width of weight and bias are the same, so no need to differenciate
    output sram1_en,
    output sram2_en,
    output sram3_en,
    output sram4_en,

    output reg signed [WU_DW-1 : 0] lstm_wu, //segment.v
    output layer_done,         // lstm completed
    // output xt_shift_en, //segment.v
    output spad1_w_we_en, // the enable signal for the first spad
    output reg [PE_NUM-2:0] spad_w_we_en_2_32, //the enable signal for the rest of PEs
    output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_re, //paralell, the same
    output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_we,  // serial operation, so share one addr_we
    output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_a_addr_re, //paralell, the same
    // output [$clog2(SPAD_DEPTH)*PE_NUM/2-1 : 0] spad_a_addr_we_1_16, //new
    output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_a_addr_we, //new
    output [PE_NUM/2-1 : 0] spad_a_we_en_1_16, //new
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
    // output [INPUT_DW*SPAD_DEPTH -1 : 0] spad1_a_data_in,
    // output [DATA_DW*SPAD_DEPTH -1 : 0] spad2_a_data_in, 
    // output [DATA_DW*SPAD_DEPTH -1 : 0] spad3_a_data_in,
    // output [DATA_DW*SPAD_DEPTH -1 : 0] spad4_a_data_in,
    // output [DATA_DW*SPAD_DEPTH -1 : 0] spad5_a_data_in,
    // output [DATA_DW*SPAD_DEPTH -1 : 0] spad6_a_data_in,
    // output [DATA_DW*SPAD_DEPTH -1 : 0] spad7_a_data_in,
    // output [DATA_DW*SPAD_DEPTH -1 : 0] spad8_a_data_in,
    // output [DATA_DW*SPAD_DEPTH -1 : 0] spad9_a_data_in,
    // output [DATA_DW*SPAD_DEPTH -1 : 0] spad10_a_data_in,
    // output [DATA_DW*SPAD_DEPTH -1 : 0] spad11_a_data_in,
    // output [DATA_DW*SPAD_DEPTH -1 : 0] spad12_a_data_in,
    // output [DATA_DW*SPAD_DEPTH -1 : 0] spad13_a_data_in,
    // output [DATA_DW*SPAD_DEPTH -1 : 0] spad14_a_data_in,
    // output [DATA_DW*SPAD_DEPTH -1 : 0] spad15_a_data_in,
    // output [DATA_DW*SPAD_DEPTH -1 : 0] spad16_a_data_in,
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
    output reg signed [PE_NUM * DATA_DW-1: 0] hardmard_a_all,
    output reg signed [PE_NUM * DATA_DW-1: 0] hardmard_b_all,
    output reg signed [B_DW-1: 0] lstm_b,
    output reg signed [DATA_BQ_DW-1:0] out_bq, // lstm: from each PEs
    output reg signed  [SCALE_DW -1 : 0] scale,
    output reg signed [DATA_BQ_DW-1:0] out_bq2, // lstm: from each PEs
    output reg signed [SCALE_DW -1 : 0] scale2,
    output signed [2*(2*DATA_DW+SCALE_DW)-1: 0] lstm_ct_temp_out_cat,
    
    output [1:0] mult_a_crl, // 00:idle, 01:requantize     
    output [1:0] mult_b_crl,
    output [1:0] add_a_crl,
    output [1:0] add_b_crl,
    output reg [2:0] mult_int8_crl_1_16, // for xt
    output reg [2:0] mult_int8_crl_17_32, //for ht
    output mult_out_round_en,
    output pe_out_sum_a_final_en,
    output pe_out_sum_b_final_en,
    input  [SRAM8_DW-1:0]  sram_act_dout,
    output [SRAM8192_AW -1 : 0] addr_sram_act,
    output sram_act_en,
    output sram_act_we,
    output [SRAM8_DW-1:0]  sram_act_din    );// lstm-> 000:idle/reset, 001:gatess_mac, 011: transfer, 111:hold, 010: hardmard prod
    
    localparam N = 4;
    localparam idle     = 4'b0000;
    localparam load_wu = 4'b0001; // from sram
    localparam load_b = 4'b0011;
    localparam load_a   = 4'b0111; // from input signal
    localparam gates_mac = 4'b1111;
    localparam tail  = 4'b1110;
    localparam load_scale =  4'b1100;
    localparam done     = 4'b1000;    
    localparam waits =  4'b1010;
    reg [N-1 : 0] lstm_state;
    reg [N-1 : 0] lstm_state_next;

    localparam BLOCK_TIMES_LAYER1 = INPUT_SIZE/SPAD_DEPTH; //the first layer input size=hs / spad_depth
    localparam BLOCK_TIMES_LAYER2 = 2*HS/SPAD_DEPTH; // the second layer
    localparam GATE_TIMES = 4*HS/(PE_NUM/2);
    localparam TAIL_TIMES  = 38 ; //16 clk for the quantization of o_t, 16clk for the quantization of hardmard product, the rest are for the data transfer, hardmard product
    localparam Q_TIMES = 16;
    localparam Q_TIMES_HARDMARD_START  = 18; //2 clk for adder
    localparam Q_TIMES_HARDMARD = 34;    
    // control counter
    reg [$clog2(SPAD_DEPTH+1)-1 : 0] cnt_sd; // spad depth
    reg [1 : 0] cnt_wu; //0,1,2,0,1,2...
    reg [$clog2(BLOCK_TIMES_LAYER2+1)-1 : 0] cnt_bt; // choose longer one
     
    reg [$clog2(GATE_TIMES+1)-1 : 0] cnt_gt; 
    reg [$clog2(T+1)-1 : 0] cnt_t;
    reg [$clog2(TAIL_TIMES+1)-1:0] cnt_tail;

    wire is_layer0;
    wire is_reverse;
    assign is_layer0 = ((lstm_top_state == 3'd1) | (lstm_top_state == 3'd2) );  
    assign is_reverse = ((lstm_top_state == 3'd2) | (lstm_top_state == 3'd4) );

    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            lstm_state <= idle;
        else
            lstm_state <= lstm_state_next;
    end

    always @(*) begin        
        case(lstm_state)
        idle: begin
            if (layer_rdy)
                lstm_state_next = load_wu;
            else
                lstm_state_next = idle;
        end
        load_wu: begin //need 32*8 sclk, need 3 wclk
            if (cnt_wu == 2) begin
                if ((cnt_gt == 0) & (cnt_bt == 0)) lstm_state_next = load_a;
                // else if (cnt_bt == 0) lstm_state_next = load_b;
                else lstm_state_next = gates_mac;                
            end 
            else lstm_state_next = load_wu;
        end
        load_a: begin
            
            lstm_state_next = gates_mac;

        end
        gates_mac: begin
            if (cnt_sd == SPAD_DEPTH)   begin
                
                if (cnt_bt == ((is_layer0)? (BLOCK_TIMES_LAYER1-1) :(BLOCK_TIMES_LAYER2-1)) ) lstm_state_next = load_b;
                else lstm_state_next = load_wu;
            end
            else lstm_state_next = gates_mac;
        end
        load_b: begin
            if ((cnt_gt == 1) & (cnt_t == 0)) lstm_state_next = load_scale;
            else if (cnt_gt  == GATE_TIMES) lstm_state_next = tail;
            else lstm_state_next = load_wu;            
        end
        load_scale: begin
            lstm_state_next = load_wu;
        end

        tail: begin
            if (cnt_tail == TAIL_TIMES) begin
               if (cnt_t < T-1) lstm_state_next = load_wu;
               else lstm_state_next = waits;
            end
            else lstm_state_next = tail;
        end
        waits: lstm_state_next = done;
        done:begin
            lstm_state_next = idle;
        end
        default:lstm_state_next = idle;
        endcase
    end
    //control counter
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            cnt_sd <= 0;
            cnt_wu <= 0;
            cnt_gt <= 0;
            cnt_bt <= 0;
            cnt_t <= 0;
            cnt_tail <= 0;
        end
        else begin
            if (lstm_state == load_wu) begin
                cnt_wu <= (cnt_wu == 2) ? 0 : cnt_wu + 1;
            end
            else if (lstm_state == gates_mac) begin
                cnt_sd <= (cnt_sd == SPAD_DEPTH) ? 0 : cnt_sd + 1;
                if (cnt_sd == SPAD_DEPTH) begin
                    cnt_bt <= (cnt_bt == ((is_layer0)? (BLOCK_TIMES_LAYER1-1) :(BLOCK_TIMES_LAYER2-1)))? 0:cnt_bt + 1;
                    if (cnt_bt == ((is_layer0)? (BLOCK_TIMES_LAYER1-1) :(BLOCK_TIMES_LAYER2-1)))  cnt_gt <= (cnt_gt == GATE_TIMES)? 0: cnt_gt + 1;
                    else cnt_gt <= cnt_gt;
                end
                else begin
                    cnt_bt <= cnt_bt;
                    cnt_gt <= cnt_gt;
                end
            end
            else if (lstm_state == tail) begin
                cnt_tail <= (cnt_tail == TAIL_TIMES)? 0 : cnt_tail + 1;
                if (cnt_tail == TAIL_TIMES) begin
                    cnt_t <= (cnt_t == T-1)? 0 : cnt_t + 1;
                    cnt_gt <= 0;
                end
                else begin
                    cnt_t <= cnt_t;
                    cnt_gt <= cnt_gt;
                end
            end
            else if (lstm_state == idle) begin
                cnt_sd <= 0;
                cnt_wu <= 0;
                cnt_gt <= 0;
                cnt_bt <= 0;
                cnt_t <= 0;
                cnt_tail <= 0;                
            end

        end
        
    end
// load_wu, load_b, load_scale：
// output spad_w_addr_we,addr_lstm_wu, spad_w_we_en, addr_lstm_b, addr_lstm_scale, addr_sram
reg [SRAM8192_AW-1:0] addr_lstm_wu;
reg [SRAM1024_AW-1:0] addr_lstm_b;
reg [SRAM1024_AW-1:0] addr_lstm_scale;

reg [SRAM8192_AW-1:0] addr_lstm_w_origin;
reg [SRAM8192_AW-1:0] addr_lstm_u_origin;


reg [PE_NUM/2* B_DW -1:0] b_buffer;

localparam SCALE_NUM = 7;//SwSx_Sg, SuSh_Sg, SiSg_Sc, SoSc_Sh,hardsigmoid_gate_scale, hardsigmoid_ct_scale
reg [$clog2(PE_NUM+1)-1:0] cnt_loaddata; // 
wire lstm_state_load_wu, lstm_state_idle,spad_w_addr_we_end, cnt_loaddata_31, cnt_loaddata_15, cnt_gt_end;

assign lstm_state_load_wu = (lstm_state == load_wu)? 1:0;
assign lstm_state_idle = (lstm_state == idle)? 1:0;
assign spad_w_addr_we_end = (spad_w_addr_we == SPAD_DEPTH-1) ? 1:0;
assign cnt_loaddata_31 = (cnt_loaddata == PE_NUM-1) ? 1:0;
assign cnt_loaddata_15 = (cnt_loaddata == PE_NUM/2-1)? 1:0;
assign cnt_gt_end = (cnt_gt == GATE_TIMES-1)? 1:0;

assign spad1_w_we_en = (lstm_state_load_wu)? ((cnt_loaddata == 0)? 1:0):0; //combinational logic


integer i;
always @(negedge sclk or negedge rst_n) begin
    if (!rst_n) spad_w_we_en_2_32 <= 0;
    else begin
        if (lstm_state_load_wu) begin


                if (spad_w_addr_we_end) begin
                    if (cnt_loaddata == 0) spad_w_we_en_2_32[0] <= spad1_w_we_en;
                    else if (cnt_loaddata<PE_NUM-1) begin // circulant shift
                        spad_w_we_en_2_32[0] <= spad_w_we_en_2_32[PE_NUM-2];
                        for ( i = 0; i < PE_NUM-2; i=i+1) begin
                            spad_w_we_en_2_32[i+1] <= spad_w_we_en_2_32[i];
                        end
                    end
                    else begin
                        spad_w_we_en_2_32 <= 0;
                    end
                end
                else spad_w_we_en_2_32<= spad_w_we_en_2_32;

        end
        else spad_w_we_en_2_32<= 0;
    end
end

reg is_lsb;

wire signed [WU_DW-1:0] sram2_dout_lsb;
wire signed [WU_DW-1:0] sram2_dout_msb;
assign sram2_dout_lsb = sram2_dout[WU_DW-1:0];
assign sram2_dout_msb = sram2_dout[2*WU_DW-1:WU_DW];


wire signed [WU_DW-1:0] sram3_dout_lsb;
wire signed [WU_DW-1:0] sram3_dout_msb;
assign sram3_dout_lsb = sram3_dout[WU_DW-1:0];
assign sram3_dout_msb = sram3_dout[2*WU_DW-1:WU_DW];


wire signed [WU_DW-1:0] sram4_dout_lsb;
wire signed [WU_DW-1:0] sram4_dout_msb;
assign sram4_dout_lsb = sram4_dout[WU_DW-1:0];
assign sram4_dout_msb = sram4_dout[2*WU_DW-1:WU_DW];


always @(posedge sclk or negedge rst_n) begin
    if (!rst_n) is_lsb <= 0;
    else begin
        if (lstm_state_load_wu) begin
            if (cnt_loaddata == PE_NUM) is_lsb <= 0;
            else  is_lsb <= ~is_lsb;
            
        end
        else is_lsb <= 0;
    end
end
reg load_b_end;
reg load_scale_end;
always @(*) begin
    if (lstm_state_load_wu) begin
        if (is_lsb) begin
            if (is_layer0) lstm_wu = sram2_dout_lsb;
            else if ((lstm_top_state == 3'd3)|( (lstm_top_state==3'd4)&(cnt_loaddata> PE_NUM/2-1))) lstm_wu = sram3_dout_lsb;
            else if ((lstm_top_state == 3'd4)& (cnt_loaddata < PE_NUM/2)) lstm_wu = sram4_dout_lsb;
            else lstm_wu  = 0;
        end
        else begin
            if (is_layer0) lstm_wu = sram2_dout_msb;
            else if ((lstm_top_state == 3'd3)|( (lstm_top_state==3'd4)&(cnt_loaddata> PE_NUM/2-1))) lstm_wu = sram3_dout_msb;
            else if ((lstm_top_state == 3'd4)& (cnt_loaddata < PE_NUM/2)) lstm_wu = sram4_dout_msb;
            else lstm_wu  = 0;            

        end
    end
    else lstm_wu= 0;
end
// assign lstm_wu = (lstm_state_load_wu)? ((is_lsb)? sram2_dout_lsb :sram2_dout_msb ): lstm_wu;
always @(negedge sclk or negedge rst_n) begin
    if (!rst_n) begin
        spad_w_addr_we <= 0;  // next start from 0
        cnt_loaddata <= 0;
        load_b_end <= 0; //for load_b
        load_scale_end <= 0;// and load_scale
    end
    else if (lstm_state_load_wu) begin
        load_b_end <= 0; 
        load_scale_end <= 0;
        if (is_layer0) begin
            if (spad_w_addr_we_end) begin
                if (cnt_loaddata == PE_NUM) begin //end
                    spad_w_addr_we <= spad_w_addr_we;
                    cnt_loaddata <= cnt_loaddata;    //END                    
                end
                else begin
                    cnt_loaddata <= cnt_loaddata + 1; 
                    spad_w_addr_we <= 0 ;                  
                end
            end 
            else begin
                if (cnt_loaddata == PE_NUM) begin //end
                    spad_w_addr_we <= spad_w_addr_we;
                    cnt_loaddata <= cnt_loaddata;    //END                    
                end   
                else begin         
                    cnt_loaddata <= cnt_loaddata; 
                    spad_w_addr_we <= spad_w_addr_we + 1;   
                end                
            end               
        end
        else begin
            if (cnt_bt < BLOCK_TIMES_LAYER1) begin // the same with layer 1
                if (spad_w_addr_we_end) begin
                    if (cnt_loaddata == PE_NUM) begin //end
                        spad_w_addr_we <= spad_w_addr_we;
                        cnt_loaddata <= cnt_loaddata;    //END                    
                    end
                    else begin
                        cnt_loaddata <= cnt_loaddata + 1; 
                        spad_w_addr_we <= 0 ;                  
                    end
                end 
                else begin
                    if (cnt_loaddata == PE_NUM) begin //end
                        spad_w_addr_we <= spad_w_addr_we;
                        cnt_loaddata <= cnt_loaddata;    //END                    
                    end   
                    else begin         
                        cnt_loaddata <= cnt_loaddata; 
                        spad_w_addr_we <= spad_w_addr_we + 1;   
                    end                
                end                    
            end
            else begin  // only w is loaded
                if (spad_w_addr_we_end) begin
                    if (cnt_loaddata == PE_NUM/2) begin //end
                        spad_w_addr_we <= spad_w_addr_we;
                        cnt_loaddata <= cnt_loaddata;    //END                    
                    end
                    else begin
                        cnt_loaddata <= cnt_loaddata + 1; 
                        spad_w_addr_we <= 0 ;                  
                    end
                end 
                else begin
                    if (cnt_loaddata == PE_NUM/2) begin //end
                        spad_w_addr_we <= spad_w_addr_we;
                        cnt_loaddata <= cnt_loaddata;    //END                    
                    end   
                    else begin         
                        cnt_loaddata <= cnt_loaddata; 
                        spad_w_addr_we <= spad_w_addr_we + 1;   
                    end                
                end                          
            end
        end

        
    end
    else if (lstm_state == load_b) begin
        if (cnt_loaddata == PE_NUM/2) begin
            cnt_loaddata <= 0; //END 
            load_b_end <= 1;
        end
        else begin
            load_b_end <= load_b_end;
            if (!load_b_end)
                cnt_loaddata <= cnt_loaddata + 1;
            else
                cnt_loaddata <= cnt_loaddata;
        end
        load_scale_end <= 0;
        spad_w_addr_we <= 0;          
    end
    else if (lstm_state == load_scale) begin 
        if (cnt_loaddata ==  SCALE_NUM) begin//cnt_loaddata  no reset to 0 from load_b,so accumulate from b
            cnt_loaddata <= 0; //END 
            load_scale_end <= 1;
        end
        else begin
            load_scale_end <= load_scale_end;
            if (!load_scale_end) 
                cnt_loaddata <= cnt_loaddata + 1;
            else 
                cnt_loaddata <= cnt_loaddata;
        end
        spad_w_addr_we <= 0;  
        load_b_end <= 0;              
    end
    else begin
        spad_w_addr_we <= 0; //reset
        cnt_loaddata <= 0;  
        load_b_end <= 0; 
        load_scale_end <= 0;
    end
end


always @(negedge sclk or negedge rst_n) begin 
    if (!rst_n) begin
        addr_lstm_wu <= 0;
        addr_lstm_w_origin <= 0;
        addr_lstm_u_origin <= 0; 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    end
    else begin
        if (lstm_state_load_wu) begin
            if (~is_lsb) begin
            
                if (spad_w_addr_we_end) begin 
                    if (cnt_loaddata_15 ) begin // change to unit
                        if (is_layer0) begin
                            if ((cnt_gt_end) & (cnt_bt == BLOCK_TIMES_LAYER1-1)) begin
                                addr_lstm_wu <= addr_lstm_u_origin;
                                addr_lstm_w_origin <= addr_lstm_w_init; // a block has finished
                                addr_lstm_u_origin <= addr_lstm_u_origin; //unchanged 
                        
                            end
                            else begin
                                addr_lstm_wu <= addr_lstm_u_origin;
                                addr_lstm_w_origin <= addr_lstm_wu + 1;
                                addr_lstm_u_origin <= addr_lstm_u_origin; //unchanged
                            end                            
                        end
                        else begin
                            if ( cnt_bt < BLOCK_TIMES_LAYER1) begin
                                addr_lstm_wu <= addr_lstm_u_origin;
                                addr_lstm_w_origin <= addr_lstm_wu + 1;
                                addr_lstm_u_origin <= addr_lstm_u_origin; //unchanged                                 
                            end
                            else begin
                                if ((cnt_gt_end) & (cnt_bt == BLOCK_TIMES_LAYER2-1))  begin
                                    addr_lstm_wu <= addr_lstm_w_init;
                                    addr_lstm_w_origin <= addr_lstm_w_init;
                                    addr_lstm_u_origin <= addr_lstm_u_origin; //unchanged   
                                end
                                else begin
                                    addr_lstm_wu <= addr_lstm_wu + 1;
                                    addr_lstm_w_origin <= addr_lstm_wu + 1;
                                    addr_lstm_u_origin <= addr_lstm_u_origin; //unchanged                                       
                                end                                   
                            end
                        end

                    end//switch to unit
                    else if (cnt_loaddata_31) begin
                        if ((cnt_gt_end) & (cnt_bt == BLOCK_TIMES_LAYER1-1)) begin
                            addr_lstm_wu <= addr_lstm_w_origin;
                            addr_lstm_w_origin <= addr_lstm_w_origin; // unchaned
                            addr_lstm_u_origin <= addr_lstm_u_init;  // a block has finished
                    
                        end
                        else begin
                            addr_lstm_wu <= addr_lstm_w_origin;
                            addr_lstm_w_origin <= addr_lstm_w_origin; // unchaned
                            addr_lstm_u_origin <= addr_lstm_wu + 1;
                        end  
                    end
                    else if (cnt_loaddata == PE_NUM) begin //END
                        addr_lstm_wu <= addr_lstm_wu;
                        addr_lstm_w_origin <= addr_lstm_w_origin; // unchaned
                        addr_lstm_u_origin <= addr_lstm_u_origin;
               
                    end //end of load unit
                    else if (cnt_loaddata == PE_NUM/2) begin
                        if (is_layer0) begin
                            addr_lstm_wu <= addr_lstm_wu + 1;
                            addr_lstm_w_origin <= addr_lstm_w_origin; // unchaned
                            addr_lstm_u_origin <= addr_lstm_u_origin; //unchanged                                
                        end
                        else begin
                            if (cnt_bt > BLOCK_TIMES_LAYER1-1) begin
                                addr_lstm_wu <= addr_lstm_wu;
                                addr_lstm_w_origin <= addr_lstm_w_origin; // unchaned
                                addr_lstm_u_origin <= addr_lstm_u_origin; 
                            end   
                            else begin
                                addr_lstm_wu <= addr_lstm_wu + 1;
                                addr_lstm_w_origin <= addr_lstm_w_origin; // unchaned
                                addr_lstm_u_origin <= addr_lstm_u_origin; //unchanged                                   
                            end                        
                        end
                    end
                    else begin

                        addr_lstm_wu <= addr_lstm_wu + 1;
                        addr_lstm_w_origin <= addr_lstm_w_origin; // unchaned
                        addr_lstm_u_origin <= addr_lstm_u_origin; //unchanged                               
                    end
                end
                else begin
                    if (cnt_loaddata == PE_NUM) begin
                        addr_lstm_wu <= addr_lstm_wu;
                        addr_lstm_w_origin <= addr_lstm_w_origin; // unchaned
                        addr_lstm_u_origin <= addr_lstm_u_origin;                        
                    end
                    else begin
                        
                        if ((cnt_loaddata == PE_NUM/2) & (cnt_bt > BLOCK_TIMES_LAYER1-1))begin
                            addr_lstm_wu <= addr_lstm_wu;
                            addr_lstm_w_origin <= addr_lstm_w_origin; // unchaned
                            addr_lstm_u_origin <= addr_lstm_u_origin;                                
                        end
                        else begin
                            addr_lstm_wu <= addr_lstm_wu + 1;
                            addr_lstm_w_origin <= addr_lstm_w_origin; // unchaned
                            addr_lstm_u_origin <= addr_lstm_u_origin; //unchanged                                 
                        end                     
                    end

                end
            end        
        end
        else if (lstm_state_idle) begin
            addr_lstm_wu <= addr_lstm_w_init;
            addr_lstm_w_origin <= addr_lstm_w_init;
            addr_lstm_u_origin <= addr_lstm_u_init;             
        end
        else ;
    end
end

//addr_lstm_scale
always @(negedge sclk or negedge rst_n) begin
    if (!rst_n) begin
        addr_lstm_scale <= 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    end
    else begin
        if (lstm_state == load_scale) begin
            if (cnt_loaddata == SCALE_NUM)
                addr_lstm_scale <= addr_lstm_scale;
            else begin
                if(!load_scale_end) addr_lstm_scale <= addr_lstm_scale + 1;   
                else         addr_lstm_scale <= addr_lstm_scale;   
            end
        end
        else if (lstm_state_idle) addr_lstm_scale <= addr_lstm_scales_init;
        else ;
    end
end

always @(*) begin
    if (lstm_state == load_b) addr_sram = {3'B0,addr_lstm_b};
    else if (lstm_state_load_wu) addr_sram = addr_lstm_wu;
    else if (lstm_state == load_scale) addr_sram = {3'B0,addr_lstm_scale};
    else addr_sram = 0;
end
assign sram1_en  = ((lstm_state == load_scale) | (lstm_state == load_b))? 1:0; // scale and b are on sram 0
assign sram2_en  = ((lstm_state_load_wu)& (is_layer0))? 1:0;
assign sram3_en  = (((lstm_state_load_wu)& (lstm_top_state==3'd3)) | ((lstm_state_load_wu )& (lstm_top_state==3'd4)&(cnt_loaddata> PE_NUM/2-1)))? 1:0;
assign sram4_en  = ((lstm_state_load_wu)& (lstm_top_state==3'd4) & (cnt_loaddata < PE_NUM/2))? 1:0;
reg signed [SCALE_DW -1 : 0] SwSx_Sg;
reg signed [SCALE_DW -1 : 0] SuSh_Sg;
reg signed [SCALE_DW -1 : 0] SiSg_Sc;
reg signed [SCALE_DW -1 : 0] SoSc_Sh;
reg signed [SCALE_DW -1 : 0] hardsigmoid_ct_scale;
reg signed [SCALE_DW -1 : 0] hardsigmoid_gate_scale;
reg signed [SCALE_DW -1 : 0] gate_scale;

always @(negedge sclk or negedge rst_n) begin
    if (!rst_n) begin
        SwSx_Sg <= 0;   
        SuSh_Sg <= 0;
        SiSg_Sc <= 0;
        SoSc_Sh <= 0; 
        hardsigmoid_ct_scale <= 0;
        hardsigmoid_gate_scale <= 0;
        gate_scale <= 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    end
    else begin
        if (lstm_state == load_scale) begin
            SwSx_Sg <= ((cnt_loaddata ==  0) & ( !load_scale_end))? sram1_dout : SwSx_Sg;
            SuSh_Sg <= (cnt_loaddata ==  1)? sram1_dout : SuSh_Sg;
            SiSg_Sc <= (cnt_loaddata ==  2)? sram1_dout : SiSg_Sc;
            SoSc_Sh <= (cnt_loaddata ==  3)? sram1_dout : SoSc_Sh;
            hardsigmoid_ct_scale <= (cnt_loaddata == 4)? sram1_dout : hardsigmoid_ct_scale;
            hardsigmoid_gate_scale  <= (cnt_loaddata == 5)? sram1_dout : hardsigmoid_gate_scale;
            gate_scale  <= (cnt_loaddata ==  SCALE_NUM-1)? sram1_dout : gate_scale;
        end
        else begin
            SwSx_Sg <= SwSx_Sg;
            SuSh_Sg <= SuSh_Sg;
            SiSg_Sc <= SiSg_Sc;
            SoSc_Sh <= SoSc_Sh;
            hardsigmoid_ct_scale <= hardsigmoid_ct_scale;
            hardsigmoid_gate_scale  <= hardsigmoid_gate_scale;
            gate_scale  <= gate_scale;            
        end
    end
    
end


always @(*) begin
    if (mult_a_crl == 2'b01) begin
        scale = SwSx_Sg; 
        scale2 = SuSh_Sg;
    end
    else if ((mult_a_crl == 2'b11) & (cnt_gt == 3)) begin
        scale = gate_scale; 
        scale2 = gate_scale;       
    end
    else if ((mult_a_crl == 2'b11) & (cnt_gt == 7)) begin
        scale = SiSg_Sc; 
        scale2 = SiSg_Sc;  
    end
    else if ((mult_a_crl == 2'b11) & (lstm_state == tail)) begin
        scale = SoSc_Sh; 
        scale2 = SoSc_Sh;  
    end
    else begin
        scale = 0;
        scale2 = 0;
    end
end
reg [SRAM8192_AW -1 : 0] addr_sram_act_origin;
reg [SRAM8192_AW -1 : 0] addr_sram_act_re;
reg [SRAM8192_AW -1 : 0] addr_sram_act_we;
reg addr_sram_act_re_end;
reg addr_sram_act_we_end;
localparam ADDR_LSTM00_SRAM_ACT_INIT = T * INPUT_SIZE;
localparam ADDR_LSTM01_SRAM_ACT_INIT = T * INPUT_SIZE + T * HS;


reg [$clog2(BLOCK_TIMES_LAYER2 * SPAD_DEPTH+1)-1:0] cnt_re_sram;
reg [$clog2(2+1)-1:0] cnt_we_sram;

// reg [$clog2(SPAD_DEPTH)-1 : 0] spad1_a_addr_we;
// reg [$clog2(SPAD_DEPTH)-1 : 0] spad2_a_addr_we;
// reg [$clog2(SPAD_DEPTH)-1 : 0] spad3_a_addr_we;
// reg [$clog2(SPAD_DEPTH)-1 : 0] spad4_a_addr_we;
// reg [$clog2(SPAD_DEPTH)-1 : 0] spad5_a_addr_we;
// reg [$clog2(SPAD_DEPTH)-1 : 0] spad6_a_addr_we;
// reg [$clog2(SPAD_DEPTH)-1 : 0] spad7_a_addr_we;
// reg [$clog2(SPAD_DEPTH)-1 : 0] spad8_a_addr_we;
// reg [$clog2(SPAD_DEPTH)-1 : 0] spad9_a_addr_we;
// reg [$clog2(SPAD_DEPTH)-1 : 0] spad10_a_addr_we;
// reg [$clog2(SPAD_DEPTH)-1 : 0] spad11_a_addr_we;
// reg [$clog2(SPAD_DEPTH)-1 : 0] spad12_a_addr_we;
// reg [$clog2(SPAD_DEPTH)-1 : 0] spad13_a_addr_we;
// reg [$clog2(SPAD_DEPTH)-1 : 0] spad14_a_addr_we;
// reg [$clog2(SPAD_DEPTH)-1 : 0] spad15_a_addr_we;
// reg [$clog2(SPAD_DEPTH)-1 : 0] spad16_a_addr_we;

reg spad1_a_we_en;
reg spad2_a_we_en;
reg spad3_a_we_en;
reg spad4_a_we_en;
reg spad5_a_we_en;
reg spad6_a_we_en;
reg spad7_a_we_en;
reg spad8_a_we_en;
reg spad9_a_we_en;
reg spad10_a_we_en;
reg spad11_a_we_en;
reg spad12_a_we_en;
reg spad13_a_we_en;
reg spad14_a_we_en;
reg spad15_a_we_en;
reg spad16_a_we_en;

// assign spad_a_addr_we_1_16 = {spad16_a_addr_we, spad15_a_addr_we, spad14_a_addr_we, spad13_a_addr_we,
//                               spad12_a_addr_we, spad11_a_addr_we, spad10_a_addr_we, spad9_a_addr_we,
//                               spad8_a_addr_we,  spad7_a_addr_we, spad6_a_addr_we, spad5_a_addr_we,
//                               spad4_a_addr_we, spad3_a_addr_we, spad2_a_addr_we, spad1_a_addr_we};
assign spad_a_we_en_1_16 = {spad16_a_we_en, spad15_a_we_en, spad14_a_we_en, spad13_a_we_en,
                            spad12_a_we_en, spad11_a_we_en, spad10_a_we_en, spad9_a_we_en,
                            spad8_a_we_en, spad7_a_we_en, spad6_a_we_en, spad5_a_we_en,
                            spad4_a_we_en, spad3_a_we_en, spad2_a_we_en, spad1_a_we_en};


always @(negedge sclk or negedge rst_n) begin
    if (!rst_n) begin
        addr_sram_act_origin <= ADDR_ENCODER_SRAM_ACT_INIT;
        addr_sram_act_re_end <= 0;
        addr_sram_act_re <= ADDR_ENCODER_SRAM_ACT_INIT;
        cnt_re_sram  <= 0;
    end
    else begin
        if (lstm_top_state == 3'd0) begin //rst
            addr_sram_act_origin <= ADDR_ENCODER_SRAM_ACT_INIT; //change
            addr_sram_act_re_end <= 0;
            addr_sram_act_re <= ADDR_ENCODER_SRAM_ACT_INIT;   
            cnt_re_sram <= 0;         
            // addr_sram_act_re <= ADDR_LSTM00_SRAM_ACT_INIT; //need to init for lstm01
            // addr_sram_act_re_end <= 0;
            // cnt_re_sram <= 0;
            // addr_sram_act_origin <=ADDR_LSTM01_SRAM_ACT_INIT+ HS * (T - 1); //need to init for lstm01   
        end
        else if (lstm_top_state == 3'd1) begin
            if (lstm_state == load_a) begin
                if (!addr_sram_act_re_end) begin
                    if ((cnt_re_sram != BLOCK_TIMES_LAYER1 * SPAD_DEPTH -1) ) begin 
                        addr_sram_act_re <= addr_sram_act_re + T;
                        addr_sram_act_re_end <= 0;
                        addr_sram_act_origin <= addr_sram_act_origin;
                        cnt_re_sram <= cnt_re_sram + 1;
                    end
                    else begin //round end
                        cnt_re_sram <= 0;
                        addr_sram_act_re <= addr_sram_act_origin + 1;
                        addr_sram_act_re_end <= 1;
                        addr_sram_act_origin <= addr_sram_act_origin + 1;                    
                    end
                end
                else begin
                    cnt_re_sram <= 0;
                    addr_sram_act_re <= addr_sram_act_re;
                    addr_sram_act_re_end <= addr_sram_act_re_end;
                    addr_sram_act_origin <= addr_sram_act_origin;
                end

            end
            else if (layer_done) begin
                addr_sram_act_re <= T -1; // 
                addr_sram_act_re_end <= 0;
                cnt_re_sram <= 0;
                addr_sram_act_origin <= T - 1; //need to init for lstm01               
            end
            else begin
                cnt_re_sram <= 0;
                addr_sram_act_re <= addr_sram_act_re;
                addr_sram_act_re_end <= 0;
                addr_sram_act_origin <= addr_sram_act_origin;                
            end
        end
        else if (lstm_top_state == 3'd2) begin
            if (lstm_state == load_a) begin
                if (!addr_sram_act_re_end) begin
                    if ((cnt_re_sram != BLOCK_TIMES_LAYER1 * SPAD_DEPTH -1) ) begin 
                        addr_sram_act_re <= addr_sram_act_re + T;
                        addr_sram_act_re_end <= 0;
                        addr_sram_act_origin <= addr_sram_act_origin;
                        cnt_re_sram <= cnt_re_sram + 1;
                    end
                    else begin //round end
                        cnt_re_sram <= 0;
                        addr_sram_act_re <= addr_sram_act_origin - 1;
                        addr_sram_act_re_end <= 1;
                        addr_sram_act_origin <= addr_sram_act_origin - 1;                    
                    end
                end
                else begin
                    cnt_re_sram <= 0;
                    addr_sram_act_re <= addr_sram_act_re;
                    addr_sram_act_re_end <= addr_sram_act_re_end;
                    addr_sram_act_origin <= addr_sram_act_origin;
                end

            end
            else if (layer_done) begin
                addr_sram_act_re <= ADDR_LSTM00_SRAM_ACT_INIT; //need to init for lstm01
                addr_sram_act_re_end <= 0;
                cnt_re_sram <= 0;
                addr_sram_act_origin <=ADDR_LSTM01_SRAM_ACT_INIT+ HS * (T - 1); //need to init for lstm01               
            end
            else begin
                cnt_re_sram <= 0;
                addr_sram_act_re <= addr_sram_act_re;
                addr_sram_act_re_end <= 0;
                addr_sram_act_origin <= addr_sram_act_origin;                
            end            
        end
        else if (lstm_top_state == 3'd3) begin
            if (lstm_state == load_a) begin
                if (!addr_sram_act_re_end) begin
                    if ((cnt_re_sram < BLOCK_TIMES_LAYER2 * SPAD_DEPTH/2 -1) ) begin 
                        addr_sram_act_re <= addr_sram_act_re + 1;
                        addr_sram_act_re_end <= 0;
                        addr_sram_act_origin <= addr_sram_act_origin;
                        cnt_re_sram <= cnt_re_sram + 1;
                    end
                    else if ((cnt_re_sram ==  BLOCK_TIMES_LAYER2 * SPAD_DEPTH/2 -1))begin  //end
                        addr_sram_act_re <= addr_sram_act_origin ;
                        addr_sram_act_re_end <= 0;
                        addr_sram_act_origin <= addr_sram_act_re + 1;
                        cnt_re_sram <= cnt_re_sram + 1;                        
                    end
                    else if ((cnt_re_sram > BLOCK_TIMES_LAYER2 * SPAD_DEPTH/2 -1) & (cnt_re_sram < BLOCK_TIMES_LAYER2 * SPAD_DEPTH -1)) begin
                        addr_sram_act_re <= addr_sram_act_re + 1;
                        addr_sram_act_re_end <= 0;
                        addr_sram_act_origin <= addr_sram_act_origin;
                        cnt_re_sram <= cnt_re_sram + 1;                        
                    end
                    else begin //round end
                        cnt_re_sram <= 0;
                        addr_sram_act_re <= addr_sram_act_origin ;
                        addr_sram_act_re_end <= 1;
                        addr_sram_act_origin <= addr_sram_act_re + 1 -2* HS;                    
                    end
                end
                else begin
                    cnt_re_sram <= 0;
                    addr_sram_act_re <= addr_sram_act_re;
                    addr_sram_act_re_end <= addr_sram_act_re_end;
                    addr_sram_act_origin <= addr_sram_act_origin;
                end

            end
            else if (layer_done) begin
                addr_sram_act_re <= ADDR_LSTM00_SRAM_ACT_INIT + HS * (T - 1) ; //need to init for lstm01
                addr_sram_act_re_end <= 0;
                cnt_re_sram <= 0;
                addr_sram_act_origin <= ADDR_LSTM01_SRAM_ACT_INIT; //need to init for lstm01               
            end
            else begin
                cnt_re_sram <= 0;
                addr_sram_act_re <= addr_sram_act_re;
                addr_sram_act_re_end <= 0;
                addr_sram_act_origin <= addr_sram_act_origin;                
            end              
        end
        else if (lstm_top_state == 3'd4) begin
            if (lstm_state == load_a) begin
                if (!addr_sram_act_re_end) begin
                    if ((cnt_re_sram < BLOCK_TIMES_LAYER2 * SPAD_DEPTH/2 -1) ) begin 
                        addr_sram_act_re <= addr_sram_act_re + 1;
                        addr_sram_act_re_end <= 0;
                        addr_sram_act_origin <= addr_sram_act_origin;
                        cnt_re_sram <= cnt_re_sram + 1;
                    end
                    else if ((cnt_re_sram ==  BLOCK_TIMES_LAYER2 * SPAD_DEPTH/2 -1))begin  //end
                        addr_sram_act_re <= addr_sram_act_origin ;
                        addr_sram_act_re_end <= 0;
                        addr_sram_act_origin <= addr_sram_act_re + 1 -2* HS;
                        cnt_re_sram <= cnt_re_sram + 1;                        
                    end
                    else if ((cnt_re_sram > BLOCK_TIMES_LAYER2 * SPAD_DEPTH/2 -1) & (cnt_re_sram < BLOCK_TIMES_LAYER2 * SPAD_DEPTH -1)) begin
                        addr_sram_act_re <= addr_sram_act_re + 1;
                        addr_sram_act_re_end <= 0;
                        addr_sram_act_origin <= addr_sram_act_origin;
                        cnt_re_sram <= cnt_re_sram + 1;                        
                    end
                    else begin //round end
                        cnt_re_sram <= 0;
                        addr_sram_act_re <= addr_sram_act_origin ;
                        addr_sram_act_re_end <= 1;
                        addr_sram_act_origin <= addr_sram_act_re + 1;                    
                    end
                end
                else begin
                    cnt_re_sram <= 0;
                    addr_sram_act_re <= addr_sram_act_re;
                    addr_sram_act_re_end <= addr_sram_act_re_end;
                    addr_sram_act_origin <= addr_sram_act_origin;
                end

            end
            else if (layer_done) begin
                addr_sram_act_origin <= ADDR_ENCODER_SRAM_ACT_INIT; //change
                addr_sram_act_re_end <= 0;
                addr_sram_act_re <= ADDR_ENCODER_SRAM_ACT_INIT;   
                cnt_re_sram <= 0;                 
            end
            else begin
                cnt_re_sram <= 0;
                addr_sram_act_re <= addr_sram_act_re;
                addr_sram_act_re_end <= 0;
                addr_sram_act_origin <= addr_sram_act_origin;                
            end              
        end
    end
end
wire lstm_hidden_unit_vld_half;
wire lstm_hidden_unit_vld;
wire signed [2*DATA_DW-1: 0] lstm_hidden_cat;
reg lstm_hidden_unit_vld_d;
reg signed [2*DATA_DW-1: 0] lstm_hidden_cat_reg;

assign lstm_hidden_unit_vld_half = wclk & lstm_hidden_unit_vld_d;

always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) lstm_hidden_unit_vld_d <= 0;
    else lstm_hidden_unit_vld_d <= lstm_hidden_unit_vld;
end 
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) lstm_hidden_cat_reg <= 0;
    else lstm_hidden_cat_reg <= lstm_hidden_cat;
end
always @(negedge sclk or negedge rst_n) begin
    if (!rst_n) begin
        addr_sram_act_we <= ADDR_LSTM00_SRAM_ACT_INIT; //change
        addr_sram_act_we_end <= 0;
        cnt_we_sram <= 0;
    end
    else begin
        if (lstm_hidden_unit_vld_half ) begin
            if (!addr_sram_act_we_end) begin
                if (cnt_we_sram == 1) begin
                    addr_sram_act_we <= addr_sram_act_we + 1;
                    addr_sram_act_we_end <= 1;
                    cnt_we_sram <= 0;
                end
                else begin
                    addr_sram_act_we <= addr_sram_act_we + 1;
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
            addr_sram_act_we_end <= 0;
            cnt_we_sram <= 0;  
            if (lstm_top_state == 3'd1) 
                addr_sram_act_we <= ADDR_LSTM01_SRAM_ACT_INIT;
            else if (lstm_top_state == 3'd2)
                addr_sram_act_we <= ADDR_LSTM10_SRAM_ACT_INIT;
            else if (lstm_top_state == 3'd3)
                addr_sram_act_we <= ADDR_LSTM11_SRAM_ACT_INIT;
            else if (lstm_top_state == 3'd4)
                addr_sram_act_we <= ADDR_LSTM00_SRAM_ACT_INIT; 
            else addr_sram_act_we <= addr_sram_act_we;
        end
        else begin
            addr_sram_act_we <= addr_sram_act_we;
            addr_sram_act_we_end <= 0;
            cnt_we_sram <= 0;                
        end
    end
end


assign sram_act_din = (lstm_hidden_unit_vld_half & !addr_sram_act_we_end)? ((cnt_we_sram)? lstm_hidden_cat_reg[2*DATA_DW-1:DATA_DW] : lstm_hidden_cat_reg[DATA_DW-1:0]):0;
assign sram_act_we = (lstm_hidden_unit_vld_half & !addr_sram_act_we_end)? 1:0;
assign sram_act_en = lstm_hidden_unit_vld_half | (lstm_state == load_a);
assign addr_sram_act = sram_act_we? addr_sram_act_we:addr_sram_act_re;
///这里的spad1_a_addr_we-spad16_a_addr_we是否可以化简，不需要PE个写地址，而是通过PE个写使能控制。
always @(negedge sclk or negedge rst_n) begin
    if (!rst_n) begin
        spad_a_addr_we <= 0;
    end
    else begin
        if (lstm_state == load_a) begin
            if (!addr_sram_act_re_end) begin 
                spad_a_addr_we <= (spad_a_addr_we == SPAD_DEPTH-1)? 0:spad_a_addr_we+1;
            end
            else spad_a_addr_we <= 0;
        end       
    end
end


always @(*) begin
    if ((lstm_top_state == 3'd1)|(lstm_top_state == 3'd2)) begin
        if (lstm_state == load_a) begin
            if (!addr_sram_act_re_end) begin
                if (cnt_re_sram < SPAD_DEPTH ) begin
                    spad1_a_data_sram_in = {{(INPUT_DW - SRAM8_DW){sram_act_dout[SRAM8_DW-1]}} , sram_act_dout};
                    spad1_a_we_en = 1;
                    spad5_a_data_sram_in = sram_act_dout;
                    spad5_a_we_en = 1;                    
                    spad9_a_data_sram_in = sram_act_dout;
                    spad9_a_we_en = 1; 
                    spad13_a_data_sram_in = sram_act_dout;
                    spad13_a_we_en = 1; 
                end
                else begin
                    spad1_a_data_sram_in = 0;
                    spad1_a_we_en = 0;
                    spad5_a_data_sram_in = 0;
                    spad5_a_we_en = 0;                    
                    spad9_a_data_sram_in = 0;
                    spad9_a_we_en = 0; 
                    spad13_a_data_sram_in = 0;
                    spad13_a_we_en = 0;                    
                end
            end
            else begin
                spad1_a_data_sram_in = 0;
                spad1_a_we_en = 0;
                spad5_a_data_sram_in = 0;
                spad5_a_we_en = 0;                    
                spad9_a_data_sram_in = 0;
                spad9_a_we_en = 0; 
                spad13_a_data_sram_in = 0;
                spad13_a_we_en = 0;                
            end
        end
        else begin
            spad1_a_data_sram_in = 0;
            spad1_a_we_en = 0;
            spad5_a_data_sram_in = 0;
            spad5_a_we_en = 0;                    
            spad9_a_data_sram_in = 0;
            spad9_a_we_en = 0; 
            spad13_a_data_sram_in = 0;
            spad13_a_we_en = 0;              
        end
    end
    else if ((lstm_top_state == 3'd3)|(lstm_top_state == 3'd4)) begin
        if (lstm_state == load_a) begin
            if (!addr_sram_act_re_end) begin
                if (cnt_re_sram < SPAD_DEPTH ) begin
                    spad1_a_data_sram_in = {{(INPUT_DW - SRAM8_DW){sram_act_dout[SRAM8_DW-1]}} , sram_act_dout};
                    spad1_a_we_en = 1;
                    spad5_a_data_sram_in = 0;
                    spad5_a_we_en = 0;                    
                    spad9_a_data_sram_in = sram_act_dout;
                    spad9_a_we_en = 1; 
                    spad13_a_data_sram_in = 0;
                    spad13_a_we_en = 0; 
                end
                else if ((cnt_re_sram >=  4*SPAD_DEPTH) & (cnt_re_sram < 5*SPAD_DEPTH)) begin
                    spad1_a_data_sram_in = 0;
                    spad1_a_we_en = 0;
                    spad5_a_data_sram_in = sram_act_dout;
                    spad5_a_we_en = 1;                    
                    spad9_a_data_sram_in = 0;
                    spad9_a_we_en = 0; 
                    spad13_a_data_sram_in = sram_act_dout;
                    spad13_a_we_en = 1;                     
                end
                else begin
                    spad1_a_data_sram_in = 0;
                    spad1_a_we_en = 0;
                    spad5_a_data_sram_in = 0;
                    spad5_a_we_en = 0;                    
                    spad9_a_data_sram_in = 0;
                    spad9_a_we_en = 0; 
                    spad13_a_data_sram_in = 0;
                    spad13_a_we_en = 0;                    
                end
            end
            else begin
                spad1_a_data_sram_in = 0;
                spad1_a_we_en = 0;
                spad5_a_data_sram_in = 0;
                spad5_a_we_en = 0;                    
                spad9_a_data_sram_in = 0;
                spad9_a_we_en = 0; 
                spad13_a_data_sram_in = 0;
                spad13_a_we_en = 0;                 
            end
        end  
        else begin
            spad1_a_data_sram_in = 0;
            spad1_a_we_en = 0;
            spad5_a_data_sram_in = 0;
            spad5_a_we_en = 0;                    
            spad9_a_data_sram_in = 0;
            spad9_a_we_en = 0; 
            spad13_a_data_sram_in = 0;
            spad13_a_we_en = 0;              
        end     
    end
    else begin
        spad1_a_data_sram_in = 0;
        spad1_a_we_en = 0;
        spad5_a_data_sram_in = 0;
        spad5_a_we_en = 0;                    
        spad9_a_data_sram_in = 0;
        spad9_a_we_en = 0; 
        spad13_a_data_sram_in = 0;
        spad13_a_we_en = 0;           
    end
end
always @(*) begin
    if ((lstm_top_state == 3'd1)|(lstm_top_state == 3'd2)) begin
        if (lstm_state == load_a) begin
            if (!addr_sram_act_re_end) begin
                if ((cnt_re_sram >=  SPAD_DEPTH) & (cnt_re_sram < 2*SPAD_DEPTH) ) begin
                    spad2_a_data_sram_in = sram_act_dout;
                    spad2_a_we_en = 1;
                    spad6_a_data_sram_in = sram_act_dout;
                    spad6_a_we_en = 1;                    
                    spad10_a_data_sram_in = sram_act_dout;
                    spad10_a_we_en = 1; 
                    spad14_a_data_sram_in = sram_act_dout;
                    spad14_a_we_en = 1; 
                end
                else begin
                    spad2_a_data_sram_in = 0;
                    spad2_a_we_en = 0;
                    spad6_a_data_sram_in = 0;
                    spad6_a_we_en = 0;                    
                    spad10_a_data_sram_in = 0;
                    spad10_a_we_en = 0; 
                    spad14_a_data_sram_in = 0;
                    spad14_a_we_en = 0;                    
                end
            end
            else begin
                spad2_a_data_sram_in = 0;
                spad2_a_we_en = 0;
                spad6_a_data_sram_in = 0;
                spad6_a_we_en = 0;                    
                spad10_a_data_sram_in = 0;
                spad10_a_we_en = 0; 
                spad14_a_data_sram_in = 0;
                spad14_a_we_en = 0;                
            end
        end
        else begin
            spad2_a_data_sram_in = 0;
            spad2_a_we_en = 0;
            spad6_a_data_sram_in = 0;
            spad6_a_we_en = 0;                    
            spad10_a_data_sram_in = 0;
            spad10_a_we_en = 0; 
            spad14_a_data_sram_in = 0;
            spad14_a_we_en = 0;             
        end
    end
    else if ((lstm_top_state == 3'd3)|(lstm_top_state == 3'd4)) begin
        if (lstm_state == load_a) begin
            if (!addr_sram_act_re_end) begin
                if ((cnt_re_sram >=  SPAD_DEPTH) & (cnt_re_sram < 2*SPAD_DEPTH) ) begin
                    spad2_a_data_sram_in = sram_act_dout;
                    spad2_a_we_en = 1;
                    spad6_a_data_sram_in = 0;
                    spad6_a_we_en = 0;                    
                    spad10_a_data_sram_in = sram_act_dout;
                    spad10_a_we_en = 1; 
                    spad14_a_data_sram_in = 0;
                    spad14_a_we_en = 0; 
                end
                else if ((cnt_re_sram >=  5*SPAD_DEPTH) & (cnt_re_sram < 6*SPAD_DEPTH)) begin
                    spad2_a_data_sram_in = 0;
                    spad2_a_we_en = 0;
                    spad6_a_data_sram_in = sram_act_dout;
                    spad6_a_we_en = 1;                    
                    spad10_a_data_sram_in = 0;
                    spad10_a_we_en = 0; 
                    spad14_a_data_sram_in = sram_act_dout;
                    spad14_a_we_en = 1;                    
                end
                else begin
                    spad2_a_data_sram_in = 0;
                    spad2_a_we_en = 0;
                    spad6_a_data_sram_in = 0;
                    spad6_a_we_en = 0;                    
                    spad10_a_data_sram_in = 0;
                    spad10_a_we_en = 0; 
                    spad14_a_data_sram_in = 0;
                    spad14_a_we_en = 0;                       
                end
            end
            else begin
                spad2_a_data_sram_in = 0;
                spad2_a_we_en = 0;
                spad6_a_data_sram_in = 0;
                spad6_a_we_en = 0;                    
                spad10_a_data_sram_in = 0;
                spad10_a_we_en = 0; 
                spad14_a_data_sram_in = 0;
                spad14_a_we_en = 0;                
            end
        end 
        else begin
            spad2_a_data_sram_in = 0;
            spad2_a_we_en = 0;
            spad6_a_data_sram_in = 0;
            spad6_a_we_en = 0;                    
            spad10_a_data_sram_in = 0;
            spad10_a_we_en = 0; 
            spad14_a_data_sram_in = 0;
            spad14_a_we_en = 0;              
        end       
    end
    else begin
        spad2_a_data_sram_in = 0;
        spad2_a_we_en = 0;
        spad6_a_data_sram_in = 0;
        spad6_a_we_en = 0;                    
        spad10_a_data_sram_in = 0;
        spad10_a_we_en = 0; 
        spad14_a_data_sram_in = 0;
        spad14_a_we_en = 0;         
    end
end
always @(*) begin
    if ((lstm_top_state == 3'd1)|(lstm_top_state == 3'd2)) begin
        if (lstm_state == load_a) begin
            if (!addr_sram_act_re_end) begin
                if ((cnt_re_sram >=  2*SPAD_DEPTH) & (cnt_re_sram < 3*SPAD_DEPTH)  ) begin
                    spad3_a_data_sram_in = sram_act_dout;
                    spad3_a_we_en = 1;
                    spad7_a_data_sram_in = sram_act_dout;
                    spad7_a_we_en = 1;                    
                    spad11_a_data_sram_in = sram_act_dout;
                    spad11_a_we_en = 1; 
                    spad15_a_data_sram_in = sram_act_dout;
                    spad15_a_we_en = 1; 
                end
                else begin
                    spad3_a_data_sram_in = 0;
                    spad3_a_we_en = 0;
                    spad7_a_data_sram_in = 0;
                    spad7_a_we_en = 0;                    
                    spad11_a_data_sram_in = 0;
                    spad11_a_we_en = 0; 
                    spad15_a_data_sram_in = 0;
                    spad15_a_we_en = 0;                    
                end
            end
            else begin
                spad3_a_data_sram_in = 0;
                spad3_a_we_en = 0;
                spad7_a_data_sram_in = 0;
                spad7_a_we_en = 0;                    
                spad11_a_data_sram_in = 0;
                spad11_a_we_en = 0; 
                spad15_a_data_sram_in = 0;
                spad15_a_we_en = 0;                
            end
        end
        else begin
            spad3_a_data_sram_in = 0;
            spad3_a_we_en = 0;
            spad7_a_data_sram_in = 0;
            spad7_a_we_en = 0;                    
            spad11_a_data_sram_in = 0;
            spad11_a_we_en = 0; 
            spad15_a_data_sram_in = 0;
            spad15_a_we_en = 0;             
        end
    end
    else if ((lstm_top_state == 3'd3)|(lstm_top_state == 3'd4)) begin
        if (lstm_state == load_a) begin
            if (!addr_sram_act_re_end) begin
                if ((cnt_re_sram >=  2*SPAD_DEPTH) & (cnt_re_sram < 3*SPAD_DEPTH)  ) begin
                    spad3_a_data_sram_in = sram_act_dout;
                    spad3_a_we_en = 1;
                    spad7_a_data_sram_in = 0;
                    spad7_a_we_en = 0;                    
                    spad11_a_data_sram_in = sram_act_dout;
                    spad11_a_we_en = 1; 
                    spad15_a_data_sram_in = 0;
                    spad15_a_we_en = 0; 
                end
                else if ((cnt_re_sram >=  6*SPAD_DEPTH) & (cnt_re_sram < 7*SPAD_DEPTH) ) begin
                    spad3_a_data_sram_in = 0;
                    spad3_a_we_en = 0;
                    spad7_a_data_sram_in = sram_act_dout;
                    spad7_a_we_en = 1;                    
                    spad11_a_data_sram_in = 0;
                    spad11_a_we_en = 0; 
                    spad15_a_data_sram_in = sram_act_dout;
                    spad15_a_we_en = 1;                    
                end
                else begin
                    spad3_a_data_sram_in = 0;
                    spad3_a_we_en = 0;
                    spad7_a_data_sram_in = 0;
                    spad7_a_we_en = 0;                    
                    spad11_a_data_sram_in = 0;
                    spad11_a_we_en = 0; 
                    spad15_a_data_sram_in = 0;
                    spad15_a_we_en = 0;                       
                end
            end
            else begin
                spad3_a_data_sram_in = 0;
                spad3_a_we_en = 0;
                spad7_a_data_sram_in = 0;
                spad7_a_we_en = 0;                    
                spad11_a_data_sram_in = 0;
                spad11_a_we_en = 0; 
                spad15_a_data_sram_in = 0;
                spad15_a_we_en = 0;                 
            end
        end 
        else begin
            spad3_a_data_sram_in = 0;
            spad3_a_we_en = 0;
            spad7_a_data_sram_in = 0;
            spad7_a_we_en = 0;                    
            spad11_a_data_sram_in = 0;
            spad11_a_we_en = 0; 
            spad15_a_data_sram_in = 0;
            spad15_a_we_en = 0;             
        end       
    end
    else begin
        spad3_a_data_sram_in = 0;
        spad3_a_we_en = 0;
        spad7_a_data_sram_in = 0;
        spad7_a_we_en = 0;                    
        spad11_a_data_sram_in = 0;
        spad11_a_we_en = 0; 
        spad15_a_data_sram_in = 0;
        spad15_a_we_en = 0;          
    end
end
always @(*) begin
    if ((lstm_top_state == 3'd1)|(lstm_top_state == 3'd2)) begin
        if (lstm_state == load_a) begin
            if (!addr_sram_act_re_end) begin
                if ((cnt_re_sram >=  3*SPAD_DEPTH) & (cnt_re_sram < 4*SPAD_DEPTH) ) begin
                    spad4_a_data_sram_in = sram_act_dout;
                    spad4_a_we_en = 1;
                    spad8_a_data_sram_in = sram_act_dout;
                    spad8_a_we_en = 1;                    
                    spad12_a_data_sram_in = sram_act_dout;
                    spad12_a_we_en = 1; 
                    spad16_a_data_sram_in = sram_act_dout;
                    spad16_a_we_en = 1;
                end
                else if ((cnt_re_sram >=  3*SPAD_DEPTH) & (cnt_re_sram < 4*SPAD_DEPTH)  ) begin
                    spad4_a_data_sram_in = 0;
                    spad4_a_we_en = 0;
                    spad8_a_data_sram_in = 0;
                    spad8_a_we_en = 0;                    
                    spad12_a_data_sram_in = 0;
                    spad12_a_we_en = 0; 
                    spad16_a_data_sram_in = 0;
                    spad16_a_we_en = 0;                    
                end
                else begin
                    spad4_a_data_sram_in = 0;
                    spad4_a_we_en = 0;
                    spad8_a_data_sram_in = 0;
                    spad8_a_we_en = 0;                    
                    spad12_a_data_sram_in = 0;
                    spad12_a_we_en = 0; 
                    spad16_a_data_sram_in = 0;
                    spad16_a_we_en = 0;                          
                end
            end
            else begin
                spad4_a_data_sram_in = 0;
                spad4_a_we_en = 0;
                spad8_a_data_sram_in = 0;
                spad8_a_we_en = 0;                    
                spad12_a_data_sram_in = 0;
                spad12_a_we_en = 0; 
                spad16_a_data_sram_in = 0;
                spad16_a_we_en = 0;                 
            end
        end
        else begin
            spad4_a_data_sram_in = 0;
            spad4_a_we_en = 0;
            spad8_a_data_sram_in = 0;
            spad8_a_we_en = 0;                    
            spad12_a_data_sram_in = 0;
            spad12_a_we_en = 0; 
            spad16_a_data_sram_in = 0;
            spad16_a_we_en = 0 ;           
        end
    end
    else if ((lstm_top_state == 3'd3)|(lstm_top_state == 3'd4)) begin
        if (lstm_state == load_a) begin
            if (!addr_sram_act_re_end) begin
                if ((cnt_re_sram >=  3*SPAD_DEPTH) & (cnt_re_sram < 4*SPAD_DEPTH)  ) begin
                    spad4_a_data_sram_in = sram_act_dout;
                    spad4_a_we_en = 1;
                    spad8_a_data_sram_in = 0;
                    spad8_a_we_en = 0;                    
                    spad12_a_data_sram_in = sram_act_dout;
                    spad12_a_we_en = 1; 
                    spad16_a_data_sram_in = 0;
                    spad16_a_we_en = 0;
                end
                else if ((cnt_re_sram >=  7*SPAD_DEPTH) & (cnt_re_sram < 8*SPAD_DEPTH) ) begin
                    spad4_a_data_sram_in = 0;
                    spad4_a_we_en = 0;
                    spad8_a_data_sram_in = sram_act_dout;
                    spad8_a_we_en = 1;                    
                    spad12_a_data_sram_in = 0;
                    spad12_a_we_en = 0; 
                    spad16_a_data_sram_in = sram_act_dout;
                    spad16_a_we_en = 1;                    
                end
                else begin
                    spad4_a_data_sram_in = 0;
                    spad4_a_we_en = 0;
                    spad8_a_data_sram_in = 0;
                    spad8_a_we_en = 0;                    
                    spad12_a_data_sram_in = 0;
                    spad12_a_we_en = 0; 
                    spad16_a_data_sram_in = 0;
                    spad16_a_we_en = 0;                    
                end
            end
            else begin
                spad4_a_data_sram_in = 0;
                spad4_a_we_en = 0;
                spad8_a_data_sram_in = 0;
                spad8_a_we_en = 0;                    
                spad12_a_data_sram_in = 0;
                spad12_a_we_en = 0; 
                spad16_a_data_sram_in = 0;
                spad16_a_we_en = 0;                
            end
        end 
        else begin
            spad4_a_data_sram_in = 0;
            spad4_a_we_en = 0;
            spad8_a_data_sram_in = 0;
            spad8_a_we_en = 0;                    
            spad12_a_data_sram_in = 0;
            spad12_a_we_en = 0; 
            spad16_a_data_sram_in = 0;
            spad16_a_we_en = 0;              
        end       
    end
    else begin
        spad4_a_data_sram_in = 0;
        spad4_a_we_en = 0;
        spad8_a_data_sram_in = 0;
        spad8_a_we_en = 0;                    
        spad12_a_data_sram_in = 0;
        spad12_a_we_en = 0; 
        spad16_a_data_sram_in = 0;
        spad16_a_we_en = 0;        
    end
end







// spad17_a_data_in ->spad32_a_data_in : h

//result:hidden_unit
reg signed [HS*DATA_DW-1:0] hidden_unit_pre;
assign mult_out_round_en = lstm_hidden_unit_vld;
assign lstm_hidden_cat = {mult_b_out_round, mult_a_out_round};
assign lstm_hidden_unit_vld = ((lstm_state == tail)&(cnt_tail > Q_TIMES_HARDMARD_START+4))? 1:0; // initialize 
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        hidden_unit_pre  <= 0; // need to change
    end 
    else begin
        if (lstm_top_state == idle)  hidden_unit_pre <= 0;
        else begin
            if (lstm_hidden_unit_vld) //ht
                hidden_unit_pre <= { lstm_hidden_cat, hidden_unit_pre[HS*DATA_DW-1:2*DATA_DW] }; //may be  need change
            else
                hidden_unit_pre <=  hidden_unit_pre;            
        end


    end
end



assign spad17_a_data_in [DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH] = hidden_unit_pre[DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH];
assign spad21_a_data_in [DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH] = hidden_unit_pre[DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH];
assign spad25_a_data_in [DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH] = hidden_unit_pre[DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH];
assign spad29_a_data_in [DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH] = hidden_unit_pre[DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH];
assign spad18_a_data_in [DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH] = hidden_unit_pre[2*DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH];
assign spad22_a_data_in [DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH] = hidden_unit_pre[2*DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH];
assign spad26_a_data_in [DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH] = hidden_unit_pre[2*DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH];
assign spad30_a_data_in [DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH] = hidden_unit_pre[2*DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH];
assign spad19_a_data_in [DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH] = hidden_unit_pre[3*DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH];
assign spad23_a_data_in [DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH] = hidden_unit_pre[3*DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH];
assign spad27_a_data_in [DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH] = hidden_unit_pre[3*DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH];
assign spad31_a_data_in [DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH] = hidden_unit_pre[3*DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH];
assign spad20_a_data_in [DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH] = hidden_unit_pre[4*DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH];
assign spad24_a_data_in [DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH] = hidden_unit_pre[4*DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH];
assign spad28_a_data_in [DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH] = hidden_unit_pre[4*DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH];
assign spad32_a_data_in [DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH] = hidden_unit_pre[4*DATA_DW*SPAD_DEPTH-1-:DATA_DW*SPAD_DEPTH];


// gate_mac:spad_w_addr_re, spad_a_addr_re
always@(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        spad_w_addr_re <= 0 ;
        spad_a_addr_re <= 0;
    end
    else begin
        if (lstm_state == gates_mac) begin
            spad_w_addr_re <= (spad_w_addr_re == SPAD_DEPTH-1)? spad_w_addr_re: spad_w_addr_re + 1;
            spad_a_addr_re <= (spad_a_addr_re == SPAD_DEPTH-1)? spad_a_addr_re:spad_a_addr_re + 1;
        end
        else begin
            spad_w_addr_re <= 0;
            spad_a_addr_re <= 0;
        end
    end
end
// assign spad_w_addr_re = cnt_sd;
// assign spad_a_addr_re = cnt_sd;
assign layer_done = (lstm_state == done) ? 1'b1 : 1'b0;

//for PE, mult_int8
reg [2:0] mult_int8_crl_1_16_n;
wire mult_int8_1_16_transfer_en;
wire mult_int8_hardmard_en;
// assign mult_int8_transfer_en = ((lstm_state  == load_wu) & (cnt_wu == 0) & (cnt_bt < BLOCK_TIMES_LAYER1-1))? 1 :0;
// assign  mult_int8_hardmard_en = (((lstm_state  == load_wu) & (cnt_wu == 0) & (cnt_bt == BLOCK_TIMES_LAYER1-1) & ((cnt_gt == 3)|(cnt_gt == 7) )) |(cnt_tail == TAIL_TIMES ))? 1 :0;
assign mult_int8_1_16_transfer_en = ((cnt_bt < ((is_layer0)? (BLOCK_TIMES_LAYER1-1) :(BLOCK_TIMES_LAYER2-1))))? 1 :0;
assign mult_int8_hardmard_en = (( (cnt_bt == ((is_layer0)? (BLOCK_TIMES_LAYER1-1) :(BLOCK_TIMES_LAYER2-1))) & ((cnt_gt == 2)|(cnt_gt == 6) )) |(cnt_tail == Q_TIMES_HARDMARD_START ))? 1 :0;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n)
        mult_int8_crl_1_16 <= idle;
    else
        mult_int8_crl_1_16 <= mult_int8_crl_1_16_n;
end
reg hard_mard_end;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n)
        hard_mard_end <= 0;
    else
        if (mult_int8_crl_1_16 == 3'b010)
            hard_mard_end <= 1;
        else
            hard_mard_end <= 0;
end

always @(*) begin
    case (mult_int8_crl_1_16)
        3'b000: begin //idle/reset
            if (lstm_state_next == gates_mac)  
                mult_int8_crl_1_16_n = 3'b001;
            else if (mult_int8_hardmard_en) // in the tail, wait for the quantization of  o_t
                mult_int8_crl_1_16_n = 3'b010;
            else
                mult_int8_crl_1_16_n = 3'b000;
        end
        3'b001: begin //gates_mac
            if (cnt_sd  == SPAD_DEPTH) begin
                if (mult_int8_1_16_transfer_en)
                    mult_int8_crl_1_16_n = 3'b011;
                else if (mult_int8_hardmard_en)
                    mult_int8_crl_1_16_n = 3'b010;
                else 
                    mult_int8_crl_1_16_n = 3'b000;
            end
            
            else
                mult_int8_crl_1_16_n = 3'b001;
        end
        
        3'b011: begin //transfer
            mult_int8_crl_1_16_n = 3'b111;
        end

        3'b111: begin //hold
            if (lstm_state_next == gates_mac) 
                mult_int8_crl_1_16_n = 3'b001;
            else
                mult_int8_crl_1_16_n = 3'b111;
        end
        3'b010: begin //hard_mard
            if (hard_mard_end)  mult_int8_crl_1_16_n = 3'b000; 
            else mult_int8_crl_1_16_n = 3'b010; //need 2 wclk
            
        end
        default:mult_int8_crl_1_16_n = 3'b000;
    endcase
end

reg [2:0] mult_int8_crl_17_32_n;
wire mult_int8_17_32_transfer_en;

assign mult_int8_17_32_transfer_en = (cnt_bt < BLOCK_TIMES_LAYER1-1)? 1 :0;

always @(posedge wclk or negedge rst_n) begin
    if (!rst_n)
        mult_int8_crl_17_32 <= idle;
    else
        mult_int8_crl_17_32 <= mult_int8_crl_17_32_n;
end

always @(*) begin
    case (mult_int8_crl_17_32)
        3'b000: begin //idle/reset
            if (lstm_state_next == gates_mac)  
                mult_int8_crl_17_32_n = 3'b001;
            else if (mult_int8_hardmard_en) // in the tail, wait for the quantization of  o_t
                mult_int8_crl_17_32_n = 3'b010;
            else
                mult_int8_crl_17_32_n = 3'b000;
        end
        3'b001: begin //gates_mac
            if (cnt_sd  == SPAD_DEPTH) begin
                if (is_layer0) begin
                    if (mult_int8_17_32_transfer_en)
                        mult_int8_crl_17_32_n = 3'b011;
                    else if (mult_int8_hardmard_en)
                        mult_int8_crl_17_32_n = 3'b010;
                    else 
                        mult_int8_crl_17_32_n = 3'b000;                    
                end
                else begin
                    if (cnt_bt <  BLOCK_TIMES_LAYER1-1) begin
                        if (mult_int8_17_32_transfer_en)
                            mult_int8_crl_17_32_n = 3'b011;
                        else 
                            mult_int8_crl_17_32_n = 3'b111;                          
                    end
                    else   begin
                        mult_int8_crl_17_32_n = 3'b111;
                    end
                end

            end
            
            else
                mult_int8_crl_17_32_n = 3'b001;
        end
        
        3'b011: begin //transfer
            mult_int8_crl_17_32_n = 3'b111;
        end

        3'b111: begin //hold
        if (is_layer0) begin
            if (lstm_state_next == gates_mac) 
                mult_int8_crl_17_32_n = 3'b001;
            else
                mult_int8_crl_17_32_n = 3'b111;
        end
        else begin
            if (cnt_bt < BLOCK_TIMES_LAYER1) begin
                if (lstm_state_next == gates_mac) 
                    mult_int8_crl_17_32_n = 3'b001;
                else
                    mult_int8_crl_17_32_n = 3'b111;                
            end
            else begin
                if ((lstm_state == gates_mac)  & (cnt_sd  == SPAD_DEPTH)) begin
                    if (cnt_bt == BLOCK_TIMES_LAYER2-1) begin
                        if (mult_int8_hardmard_en) mult_int8_crl_17_32_n = 3'b010;
                        else mult_int8_crl_17_32_n = 3'b000;    
                    end            
                    else mult_int8_crl_17_32_n = 3'b111;
                end
                else
                    mult_int8_crl_17_32_n = 3'b111;  
            end          
        end
        end
        3'b010: begin //hard_mard
            if (hard_mard_end)  mult_int8_crl_17_32_n = 3'b000; 
            else mult_int8_crl_17_32_n = 3'b010; //need 2 wclk
            
        end
        default:mult_int8_crl_17_32_n = 3'b000;
    endcase
end
//for PE, mult_a_crl, mult_b_crl

wire q_rdy;
reg q_time;
reg [$clog2(Q_TIMES_HARDMARD+1)-1:0] cnt_q;
assign q_rdy = (lstm_state  == load_b) ;

always @(posedge wclk or negedge rst_n) begin
    if (!rst_n)
        q_time <= 0;
    else begin
        if (q_rdy) q_time <= 1;
        else begin
            if ((cnt_gt == 3) | (cnt_gt == 7) ) begin 
                if (cnt_q == Q_TIMES_HARDMARD) q_time <= 0;
                else q_time <= q_time;
            end
            else begin
                if (cnt_q == Q_TIMES+1) q_time <= 0;
                else q_time <= q_time;
            end
        end
    end     
end


always @(posedge wclk or negedge rst_n) begin
    if (!rst_n)
        cnt_q <= 0;
    else begin
        if  ((cnt_gt == 3) | (cnt_gt == 7) ) begin
            if (q_time) cnt_q <= (cnt_q == Q_TIMES_HARDMARD)? 0: cnt_q + 1;
            else  cnt_q <= cnt_q;
        end
        else begin
            if (q_time) cnt_q <= (cnt_q == Q_TIMES+1)? 0: cnt_q + 1; // for b_buffer
            else  cnt_q <= cnt_q;            
        end

    end     
end
// addr_lstm_b;
always @(negedge sclk or negedge rst_n) begin
    if (!rst_n)  addr_lstm_b <= 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    else begin
        if (lstm_state == load_b) begin
            if (cnt_loaddata == PE_NUM/2) begin
                if (cnt_gt == 8)  addr_lstm_b <= addr_lstm_b_init;                 
                else addr_lstm_b <= addr_lstm_b;     
            end
            else  begin
                if (!load_b_end) 
                    addr_lstm_b <= addr_lstm_b + 1;
                else addr_lstm_b <= addr_lstm_b ;
            end
        end
        else if (lstm_state_idle) begin
            addr_lstm_b <= addr_lstm_b_init;        
        end
        else ;
    end
end

always @(negedge sclk or negedge rst_n) begin
    if (!rst_n)
        b_buffer <= 0;
    else begin
        if ((lstm_state == load_b) &(cnt_loaddata<PE_NUM/2) & (!load_b_end)) 
            b_buffer <= {b_buffer[PE_NUM/2* B_DW - 1 : 0], sram1_dout};
            // b_buffer <= {sram1_dout , b_buffer[PE_NUM* B_DW - 1 : B_DW]};
        else
            b_buffer  <= b_buffer;
    end
end
always @(*) begin
    case(cnt_q)
        5'd2:lstm_b= b_buffer[16*B_DW-1 : 15*B_DW];
        5'd3:lstm_b= b_buffer[15*B_DW-1 : 14*B_DW];
        5'd4:lstm_b= b_buffer[14*B_DW-1 : 13*B_DW];
        5'd5:lstm_b= b_buffer[13*B_DW-1 : 12*B_DW];
        5'd6:lstm_b= b_buffer[12*B_DW-1 : 11*B_DW];
        5'd7:lstm_b= b_buffer[11*B_DW-1 : 10*B_DW];
        5'd8:lstm_b= b_buffer[10*B_DW-1 : 9*B_DW];
        5'd9:lstm_b= b_buffer[9*B_DW-1 : 8*B_DW];
        5'd10:lstm_b= b_buffer[8*B_DW-1 : 7*B_DW];
        5'd11:lstm_b= b_buffer[7*B_DW-1 : 6*B_DW];
        5'd12:lstm_b= b_buffer[6*B_DW-1 : 5*B_DW];
        5'd13:lstm_b= b_buffer[5*B_DW-1 : 4*B_DW];
        5'd14:lstm_b= b_buffer[4*B_DW-1 : 3*B_DW];
        5'd15:lstm_b= b_buffer[3*B_DW-1 : 2*B_DW];
        5'd16:lstm_b= b_buffer[2*B_DW-1 : 1*B_DW];
        5'd17:lstm_b= b_buffer[1*B_DW-1 : 0];
    default:lstm_b = 0;
    endcase
end
// assign lstm_b = b_buffer[B_DW-1 : 0];

assign mult_a_crl = (((q_time) & (cnt_q < Q_TIMES) ) |((lstm_state == tail)&(cnt_tail <= PE_NUM/2)))? 2'b01 : 
                    (((q_time) & (cnt_q > Q_TIMES_HARDMARD_START)&((cnt_gt ==3) | (cnt_gt ==7)) ) |((lstm_state == tail)&(cnt_tail >Q_TIMES_HARDMARD_START)))? 2'b11:2'b00;
assign mult_b_crl = mult_a_crl;

reg [1:0] mult_a_crl_d; // for adder_a
reg [1:0] mult_a_crl_2d; // for adder_b
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        mult_a_crl_d <= 0;
        mult_a_crl_2d <= 0;
    end
    else begin
        mult_a_crl_d <= mult_a_crl;
        mult_a_crl_2d <= mult_a_crl_d;
    end     
end
// assign add_a_crl = (q_time)? (((mult_a_crl_d  == 2'b11) & (cnt_gt == 7))? 2'b11 :mult_a_crl_d): 0;
// assign add_b_crl = (q_time)? (((mult_a_crl_d  == 2'b11) & (cnt_gt == 7))? 2'b11 :mult_a_crl_2d): 0;
assign add_a_crl = (mult_a_crl_d == 2'b01)? mult_a_crl_d:(((mult_a_crl_d  == 2'b11)& (cnt_gt == 7))?mult_a_crl_d :0);
assign add_b_crl = (mult_a_crl_2d == 2'b01)? mult_a_crl_2d:(((mult_a_crl_d  == 2'b11)& (cnt_gt == 7))?mult_a_crl_d :0);
// buffers for the storage of results from PE,
reg  [2*HS*DATA_DW-1:0] gates_buffer;
reg  [PE_NUM/2*DATA_BQ_DW-1:0] gates_x_bq_buffer;
reg  [PE_NUM/2*DATA_BQ_DW-1:0] gates_h_bq_buffer;
reg  [PE_NUM*DATA_BQ_DW/2-1:0] tail_bq_buffer;
reg  [HS*(2*DATA_DW+SCALE_DW)-1:0] ct_buffer;
reg  [HS*DATA_DW-1:0] ct_sigmoid_buffer;

// gates_bq_buffer
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        gates_x_bq_buffer <= 0;
        gates_h_bq_buffer <= 0;
    end
    else begin
        if (lstm_state_idle) begin
            gates_x_bq_buffer <= 0;
            gates_h_bq_buffer <= 0;            
        end
        else begin
            if (q_rdy)  begin
                gates_x_bq_buffer <= pe_out_32b_all[PE_NUM/2*DATA_BQ_DW-1:0]; // lower bits
                gates_h_bq_buffer <= pe_out_32b_all[PE_NUM*DATA_BQ_DW-1:PE_NUM/2*DATA_BQ_DW]; // higher bits
            end
            else if (mult_a_crl == 2'b01)  begin//requantization
                gates_x_bq_buffer <= {gates_x_bq_buffer[DATA_BQ_DW-1:0],gates_x_bq_buffer[PE_NUM/2*DATA_BQ_DW -1:DATA_BQ_DW]  };
                gates_h_bq_buffer <= {gates_h_bq_buffer[DATA_BQ_DW-1:0],gates_h_bq_buffer[PE_NUM/2*DATA_BQ_DW -1:DATA_BQ_DW]  };
            end
            else begin
                gates_x_bq_buffer <= gates_x_bq_buffer;
                gates_h_bq_buffer <= gates_h_bq_buffer;
            end            
        end

    end     
end
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_1;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_2;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_3;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_4;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_5;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_6;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_7;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_8;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_9;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_10;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_11;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_12;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_13;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_14;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_15;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_16;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_17;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_18;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_19;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_20;    
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_21;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_22;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_23;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_24;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_25;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_26;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_27;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_28;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_29;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_30;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_31;
    wire signed [DATA_BQ_DW/2-1: 0] psum_out_32b_32;

    assign psum_out_32b_1 = (is_layer0)? pe_out_32b_all[2*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2]: pe_out_32b_all[2*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_2 = (is_layer0)? pe_out_32b_all[3*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2]:pe_out_32b_all[3*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_3 = (is_layer0)? pe_out_32b_all[4*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2]:pe_out_32b_all[4*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_4 = (is_layer0)? pe_out_32b_all[1*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2]:pe_out_32b_all[5*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_5 = (is_layer0)? pe_out_32b_all[6*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2]:pe_out_32b_all[6*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_6 = (is_layer0)? pe_out_32b_all[7*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2]:pe_out_32b_all[7*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_7 = (is_layer0)? pe_out_32b_all[8*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2]:pe_out_32b_all[8*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_8 = (is_layer0)? pe_out_32b_all[5*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2]:pe_out_32b_all[1*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_9 = (is_layer0)? pe_out_32b_all[10*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2]:pe_out_32b_all[10*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_10 = (is_layer0)? pe_out_32b_all[11*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2]:pe_out_32b_all[11*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_11 = (is_layer0)? pe_out_32b_all[12*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2]:pe_out_32b_all[12*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_12 = (is_layer0)? pe_out_32b_all[9*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2]:pe_out_32b_all[13*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_13 = (is_layer0)? pe_out_32b_all[14*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2]:pe_out_32b_all[14*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_14 = (is_layer0)? pe_out_32b_all[15*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2]:pe_out_32b_all[15*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_15 = (is_layer0)? pe_out_32b_all[16*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2]:pe_out_32b_all[16*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_16 = (is_layer0)? pe_out_32b_all[13*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2]:pe_out_32b_all[9*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_17 = pe_out_32b_all[18*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_18 = pe_out_32b_all[19*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_19 = pe_out_32b_all[20*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_20 = pe_out_32b_all[17*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_21 = pe_out_32b_all[22*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_22 = pe_out_32b_all[23*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_23 = pe_out_32b_all[24*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_24 = pe_out_32b_all[21*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_25 = pe_out_32b_all[26*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_26 = pe_out_32b_all[27*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_27 = pe_out_32b_all[28*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_28 = pe_out_32b_all[25*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_29 = pe_out_32b_all[30*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_30 = pe_out_32b_all[31*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_31 = pe_out_32b_all[32*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    assign psum_out_32b_32 = pe_out_32b_all[29*DATA_BQ_DW-1-DATA_BQ_DW/2-:DATA_BQ_DW/2];
    wire [PE_NUM*DATA_BQ_DW/2-1: 0] psum_out_32b_all_for_hardmard; //no need to change order， and hardmard product only has half width
    assign  psum_out_32b_all_for_hardmard  = {psum_out_32b_32, psum_out_32b_31,psum_out_32b_30,psum_out_32b_29,
                                            psum_out_32b_28,psum_out_32b_27,psum_out_32b_26,psum_out_32b_25,
                                            psum_out_32b_24,psum_out_32b_23,psum_out_32b_22,psum_out_32b_21,
                                            psum_out_32b_20,psum_out_32b_19,psum_out_32b_18,psum_out_32b_17,
                                            psum_out_32b_16,psum_out_32b_15,psum_out_32b_14,psum_out_32b_13,
                                            psum_out_32b_12,psum_out_32b_11,psum_out_32b_10,psum_out_32b_9,
                                            psum_out_32b_8,psum_out_32b_7,psum_out_32b_6,psum_out_32b_5,
                                            psum_out_32b_4,psum_out_32b_3,psum_out_32b_2,psum_out_32b_1};

//tail_bq_buffer
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n)
        tail_bq_buffer <= 0;
    else begin
        if (lstm_state_idle) begin
            tail_bq_buffer <= 0;
        end
        else begin
            if ((((lstm_state  == load_wu) & (cnt_wu == 1) & (cnt_bt ==0) & ((cnt_gt == 3)|(cnt_gt == 7 ) )) |(cnt_tail == Q_TIMES_HARDMARD_START + 3))) // three times: ft*ct, it*gt, ot*ct
                tail_bq_buffer <= psum_out_32b_all_for_hardmard;
                // tail_bq_buffer <= 444;
            else if (mult_a_crl == 2'b11) // requantization
                tail_bq_buffer <= {tail_bq_buffer[DATA_BQ_DW-1:0],tail_bq_buffer[PE_NUM*DATA_BQ_DW/2 -1:DATA_BQ_DW]  };
            else
                tail_bq_buffer <= tail_bq_buffer;            
        end

    end     
end


// out_bq, scale, out_bq2, scale2
always @(*) begin
    if (mult_a_crl == 2'b01) begin
        out_bq = gates_x_bq_buffer[DATA_BQ_DW-1:0];
        out_bq2 = gates_h_bq_buffer[DATA_BQ_DW-1:0];        
    end
    else if (mult_a_crl == 2'b11) begin
        out_bq = {{(DATA_BQ_DW/2){tail_bq_buffer[DATA_BQ_DW/2-1]}},tail_bq_buffer[DATA_BQ_DW/2-1:0]};
        out_bq2 = {{(DATA_BQ_DW/2){tail_bq_buffer[DATA_BQ_DW-1]}},tail_bq_buffer[DATA_BQ_DW-1:DATA_BQ_DW/2]};         
    end
    else begin
        out_bq = 0;
        out_bq2 = 0;
    end
end
// gate buffer

wire signed [DATA_DW-1: 0] lstm_gate; //
wire signed [DATA_DW-1:0] hardsigmoid_gate_sum;//ft, it, ot


assign hardsigmoid_gate_sum = (hardsigmoid_gate_scale > 0)? 
                                ( (pe_out_sum_b_final > hardsigmoid_gate_scale) ? 
                                hardsigmoid_gate_scale : (pe_out_sum_b_final < -hardsigmoid_gate_scale) ? - hardsigmoid_gate_scale : pe_out_sum_b_final) :
                                ( (pe_out_sum_b_final < hardsigmoid_gate_scale) ? 
                                hardsigmoid_gate_scale : (pe_out_sum_b_final > -hardsigmoid_gate_scale) ? - hardsigmoid_gate_scale : pe_out_sum_b_final);
wire signed [DATA_DW-1:0] relu_gate_sum;
assign relu_gate_sum = (pe_out_sum_b_final > 0)? pe_out_sum_b_final :0; //gt
assign lstm_gate = ((cnt_gt == 5) | (cnt_gt == 6)) ? //gt
                  relu_gate_sum : hardsigmoid_gate_sum;  
reg [1:0]  add_b_crl_d ;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n)
        add_b_crl_d <= 0;
    else begin
        add_b_crl_d <= add_b_crl;
    end     
end
// wire gates_vld;
// assign gates_vld = (add_b_crl_d==2'B01);
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n)
        gates_buffer <= 0;
    else begin
        if (lstm_state_idle) begin
            gates_buffer<=0;
        end
        else begin
            if (add_b_crl_d==2'B01)
                gates_buffer <= {lstm_gate, gates_buffer[2*HS*DATA_DW-1:DATA_DW]}; 
            else
                gates_buffer <= gates_buffer;            
        end

    end     
end

//ct_buffer
localparam HARDMARD_PROD_OUT_DW = 2*DATA_DW+SCALE_DW;
// localparam GATES_SCALE_E = 15;
wire  [2*HARDMARD_PROD_OUT_DW-1: 0] lstm_ct_cat; //torch.clamp(torch.round(gates_scale * f_t[hs] * c_t[hs] + SiSg_Sc * i_t[hs] * g_t[hs]),-127, 127),for ct_buffer
wire signed [HARDMARD_PROD_OUT_DW-1: 0] lstm_ct_temp_a;
wire signed [HARDMARD_PROD_OUT_DW-1: 0] lstm_ct_temp_b;
wire  [2*HARDMARD_PROD_OUT_DW-1: 0] lstm_ct_temp_cat; //gates_scale * f_t[hs] * c_t[hs],to ct_buffer for save
// assign lstm_ct_temp_a = ((mult_a_crl_d == 2'B11) & (cnt_gt == 3))?(pe_out_a<<<GATES_SCALE_E):0; // need to select in lstm.v, to expand gates_scale_quan 
// assign lstm_ct_temp_b = ((mult_a_crl_d == 2'B11) & (cnt_gt == 3))?(pe_out_b<<<GATES_SCALE_E):0; // need to select in lstm.v, to expand gates_scale_quan
assign lstm_ct_temp_a = ((mult_a_crl_d == 2'B11) & (cnt_gt == 3))?(pe_out_a):0; // need to select in lstm.v, to expand gates_scale_quan 
assign lstm_ct_temp_b = ((mult_a_crl_d == 2'B11) & (cnt_gt == 3))?(pe_out_b):0; // need to select in lstm.v, to expand gates_scale_quan
assign lstm_ct_temp_cat =  {lstm_ct_temp_b, lstm_ct_temp_a};

assign lstm_ct_cat = {{(HARDMARD_PROD_OUT_DW-DATA_DW){pe_out_sum_b_final[DATA_DW-1]}},pe_out_sum_b_final,{(HARDMARD_PROD_OUT_DW-DATA_DW){pe_out_sum_a_final[DATA_DW-1]}}, pe_out_sum_a_final};

reg [1:0] add_a_crl_d;

always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        add_a_crl_d <= 0;

    end
    else begin
        add_a_crl_d <= add_a_crl;

    end
end

assign pe_out_sum_a_final_en = ((add_a_crl_d == 2'B11) & (cnt_gt == 7));
assign pe_out_sum_b_final_en = (pe_out_sum_a_final_en | (add_b_crl_d==2'B01));
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n)
        ct_buffer <= 0;
    else begin
        if (lstm_top_state == idle) begin
            ct_buffer <= 0;
        end
        else begin
            if ((mult_a_crl_d == 2'B11) & (cnt_gt == 3))
                ct_buffer <= {lstm_ct_temp_cat, ct_buffer[HS*HARDMARD_PROD_OUT_DW-1:2*HARDMARD_PROD_OUT_DW]}; 
            else if ((add_a_crl_d == 2'B11) & (cnt_gt == 7))
                ct_buffer <= {lstm_ct_cat, ct_buffer[HS*HARDMARD_PROD_OUT_DW-1:2*HARDMARD_PROD_OUT_DW]}; 
            else  
                ct_buffer <= ct_buffer;
        end

    end     
end

assign lstm_ct_temp_out_cat =  (cnt_q == Q_TIMES_HARDMARD_START+2)? ct_buffer[2*HARDMARD_PROD_OUT_DW-1:0]: ct_buffer[4*HARDMARD_PROD_OUT_DW-1:2*HARDMARD_PROD_OUT_DW] ;
//ct_sigmoid_buffer
wire signed [2*DATA_DW-1: 0] lstm_ct_sigmoid_cat; //hardsigmoid(torch.clamp(torch.round(gates_scale * f_t[hs] * c_t[hs] + SiSg_Sc * i_t[hs] * g_t[hs]),-127, 127)), for hardmard_a/b
//hardsigmoid(ct)
wire signed [DATA_DW-1:0] hardsigmoid_ct_a;  
wire signed [DATA_DW-1:0] hardsigmoid_ct_b;      
assign hardsigmoid_ct_a = (hardsigmoid_ct_scale > 0)? 
                                ( (pe_out_sum_a_final > hardsigmoid_ct_scale) ? 
                                hardsigmoid_ct_scale : (pe_out_sum_a_final < -hardsigmoid_ct_scale) ? - hardsigmoid_ct_scale : pe_out_sum_a_final) :
                                ( (pe_out_sum_a_final < hardsigmoid_ct_scale) ? 
                                hardsigmoid_ct_scale : (pe_out_sum_a_final > -hardsigmoid_ct_scale) ? - hardsigmoid_ct_scale : pe_out_sum_a_final);   
assign hardsigmoid_ct_b = (hardsigmoid_ct_scale > 0)? 
                                ( (pe_out_sum_b_final > hardsigmoid_ct_scale) ? 
                                hardsigmoid_ct_scale : (pe_out_sum_b_final < -hardsigmoid_ct_scale) ? - hardsigmoid_ct_scale : pe_out_sum_b_final) :
                                ( (pe_out_sum_b_final < hardsigmoid_ct_scale) ? 
                                hardsigmoid_ct_scale : (pe_out_sum_b_final > -hardsigmoid_ct_scale) ? - hardsigmoid_ct_scale : pe_out_sum_b_final);                                          
// wire ct_vld;
// assign ct_vld = (add_a_crl_d == 2'B11) & (cnt_gt == 7);
assign lstm_ct_sigmoid_cat =  {hardsigmoid_ct_b, hardsigmoid_ct_a };
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n)
        ct_sigmoid_buffer <= 0;
    else begin
        if (lstm_state_idle) begin
            ct_sigmoid_buffer <= 0;
        end
        else begin
            if ((add_a_crl_d == 2'B11) & (cnt_gt == 7))
                ct_sigmoid_buffer <= {lstm_ct_sigmoid_cat, ct_sigmoid_buffer[HS*DATA_DW-1:2*DATA_DW]}; 

            else  
                ct_sigmoid_buffer <= ct_sigmoid_buffer;            
        end

    end     
end
//hardmard_a_all, hardmard_b_all
integer o;
always @(*) begin
    if (cnt_gt == 3) begin
        hardmard_a_all = gates_buffer[2*HS*DATA_DW-1:HS*DATA_DW]; // HS == PE_NUM, ft
        for ( o = 0; o < HS; o=o+1) begin
            hardmard_b_all[(o+1)*DATA_DW-1-:DATA_DW] = ct_buffer[o*HARDMARD_PROD_OUT_DW+DATA_DW-1-:DATA_DW];
        end
    end
    else if (cnt_gt == 7) begin
        hardmard_a_all = gates_buffer[HS*DATA_DW-1:0]; // it
        hardmard_b_all = gates_buffer[2*HS*DATA_DW-1:HS*DATA_DW];   //gt    
    end
    else if (lstm_state == tail) begin
        hardmard_a_all = ct_sigmoid_buffer[HS*DATA_DW-1:0]; // ct_sigmoid
        hardmard_b_all = gates_buffer[2*HS*DATA_DW-1:HS*DATA_DW];   //ot    
    end
    else begin
        hardmard_a_all = 0;
        hardmard_b_all = 0;
    end
end














endmodule