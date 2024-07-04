`timescale  1ns/100ps
module ENCODER #(parameter ENCODER_WB_DW = 32,
                 SRAM8192_AW = 13, //NEW
                 SRAM8_DW = 8,
                 DATA_DW = 12,
                 SRAM_AW = 10,
                 SRAM_DW = 32,
                 DATA_OUT_DW = 8,
                 SPAD_DEPTH = 8,
                 SCALE_DW = 16,
                 STRIDE = 4,
                 LENGTH_IN = 256,
                 PADDING_PRE = 3,
                 PADDING_POST = 1,                                                                                                                                                                                                                                                                                                                                                                                                    //�?后一次卷积只�?要一个padding
                 KERNEL_SIZE = 8,
                 CHANNEL_IN = 1,                                                                                                                                                                                                                                                                                                                                                                                                      //�?后一次卷积只�?要一个padding
                 CHANNEL_OUT = 32,
                 LENGTH_OUT = 64,
                 ADDR_ENCODER_SRAM_ACT_INIT = 0)
               (input wclk,
                 input sclk,
                 input rst_n,
                 input [3:0] seg_state, 
                 input [LENGTH_IN*DATA_DW-1:0] input_signal,
                 
                 input [SRAM_AW-1:0] addr_encoder_w_init,                                                                                                                                                                                                                                                                                                                                                                             // from top.v
                 input [SRAM_AW-1:0] addr_encoder_b_init,                                                                                                                                                                                                                                                                                                                                                                             // from top.v
                 input [SRAM_AW-1:0] addr_encoder_output_scale,                                                                                                                                                                                                                                                                                                                                                                       // from top.v                                                                                                                                                                                                                                                                                                                                                                      // from top.v
                 input signed [SRAM_DW-1 : 0] sram_dout,                                                                                                                                                                                                                                                                                                                                                                                               //sram_dout
                 input encoder_rdy,
                 output reg [SRAM_AW-1:0] addr_encoder_wb,  
                 output sram1_en,                                                                                                                                                                                                                                                                                                                                                                          // data width of weight and bias are the same, so no need to differenciate
                 output encoder_done,                                                                                                                                                                                                                                                                                                                                                                                                 // encoder completed
                 output reg [1:0] shift_en,
                 output spad_w_we_en,
                 output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_re, 
                 output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_w_addr_we, 
                 output reg [$clog2(SPAD_DEPTH)-1 : 0] spad_a_addr_re, 
                 output reg [DATA_DW*SPAD_DEPTH -1 : 0] spad_a_data_in, 
                 output reg signed [ENCODER_WB_DW-1 : 0] encoder_b, 
                 output signed [ENCODER_WB_DW-1 : 0] encoder_w, 
                 output reg [1:0] mult_a_crl, 
                 output reg signed [SCALE_DW -1 : 0] scale, 
                 output relu_en,
                 output round_en,
                 output reg encoder_out_vld,
                 input signed [DATA_OUT_DW-1:0] encoder_out,
                 output reg [SRAM8192_AW - 1:0] addr_encoder_sram_act, //NEW
                 output [SRAM8_DW -1 :0]  encoder_sram_act_din, //new
                 output  encoder_sram_act_en,//NEW
                 output  encoder_sram_act_we); //NEW
    
    
    
    localparam N       = 3;
    localparam idle    = 3'b000;
    localparam load_wb = 3'b001;
    localparam load_a  = 3'b010;
    localparam mac     = 3'b011;
    localparam add_b   = 3'b101;
    localparam requant = 3'b110;
    localparam done    = 3'b111;
    localparam waits    = 3'b100;   

    reg [N-1 : 0] encoder_state;
    reg [N-1 : 0] encoder_state_next;
    
    // control counter
    reg [$clog2(KERNEL_SIZE+1)-1 : 0] cnt_ks;
    reg [$clog2(LENGTH_OUT+1)-1 : 0] cnt_lo;
    reg [$clog2(CHANNEL_OUT+1)-1 : 0] cnt_cho;
    
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n)
            encoder_state <= idle;
        else
            encoder_state <= encoder_state_next;
    end
    
    always @(*) begin
        case (encoder_state)
            idle: begin
                if (encoder_rdy)
                    encoder_state_next = load_wb;
                else
                    encoder_state_next = idle;
            end
            load_wb: begin
                encoder_state_next = load_a;
            end
            // load_a: begin
            //     if (cnt_lo == LENGTH_OUT) // need 65 times shif t to reset the act_sr
            //         encoder_state_next = load_a;
            //     else
            //         encoder_state_next = mac;
            // end
            load_a: begin
                encoder_state_next = mac;
            end            
            mac: begin
                if (cnt_ks != KERNEL_SIZE)
                    encoder_state_next = mac;
                else
                    encoder_state_next = add_b;
            end
            add_b: begin
                encoder_state_next = requant;
            end
            
            requant: begin
                if ((cnt_lo != 0))
                    encoder_state_next = load_a;
                else if ((cnt_lo == 0) & (cnt_cho != CHANNEL_OUT))
                    encoder_state_next = load_wb;
                else if ((cnt_lo == 0) & (cnt_cho == CHANNEL_OUT))
                    encoder_state_next = waits;
                end
            waits: encoder_state_next = done; // for save
            done:encoder_state_next = idle;
            default: encoder_state_next = idle;
        endcase
    end
    
    always @(*) begin
        if (encoder_state == mac)
            mult_a_crl = 2'b01;
        else if (encoder_state == add_b)
            mult_a_crl = 2'b11;
        else if (encoder_state == requant)
            mult_a_crl = 2'b10;
        else
            mult_a_crl = 2'b00;
    end
    // control counter
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            cnt_ks  <= 0;
            cnt_lo  <= 0;
            cnt_cho <= 0;
        end
        else begin
            if (encoder_state == mac) begin
                cnt_ks <= (cnt_ks == KERNEL_SIZE)? 0 : cnt_ks + 1'b1;
                cnt_lo  <= cnt_lo;
                cnt_cho <= cnt_cho;
            end
            else if (encoder_state == load_a)  begin //
                cnt_lo <= (cnt_lo == LENGTH_OUT-1)? 0 : cnt_lo + 1'b1;
                cnt_ks  <= cnt_ks;
                cnt_cho <= cnt_cho;
            end
            else if (encoder_state == load_wb)  begin
                cnt_cho <= (cnt_cho == CHANNEL_OUT)? 0 : cnt_cho + 1'b1;
                cnt_ks <= cnt_ks;
                cnt_lo <= cnt_lo;
                end
            else if (encoder_state == requant)  begin
                cnt_cho <= cnt_cho;
                cnt_ks <= cnt_ks;
                cnt_lo <= cnt_lo;
                end 
            // else if (encoder_state == requant)  begin
            //     cnt_cho <= (cnt_cho == CHANNEL_OUT-1)? 0 : cnt_cho + 1'b1;
            //     cnt_ks <= cnt_ks;
            //     cnt_lo <= (cnt_lo == LENGTH_OUT-1)? 0 : cnt_lo + 1'b1;
            //     end              
            else if (encoder_done) begin
                cnt_ks  <= 0;
                cnt_lo  <= 0;
                cnt_cho <= 0;
            end
            else begin
                cnt_ks  <= cnt_ks;
                cnt_lo  <= cnt_lo;
                cnt_cho <= cnt_cho;                
            end
        end
    end
    
    // load_wb: Outputs: addr_encoder_wb,spad_w_addr_we,spad_w_we_en,encoder_b/encoder_w (sclk), scale(sclk)
    reg [SRAM_AW-1:0] addr_encoder_w_orgin; // start address for weight of each turn
    reg [SRAM_AW-1:0] addr_encoder_b_orgin;
    reg [$clog2(KERNEL_SIZE+1)-1:0] cnt_wb;
    wire wbsz_init_en;
    always@(negedge sclk or negedge rst_n) begin
        if (!rst_n)  cnt_wb       <= 0 ;
        
        else if (encoder_state == load_wb) begin
            if (cnt_wb == KERNEL_SIZE+2) cnt_wb <= cnt_wb;
            else cnt_wb       <= cnt_wb + 1;               
        end          
        else  cnt_wb  <= 0;

    end
    
    assign wbsz_init_en = (encoder_state == load_wb)? ((cnt_wb < KERNEL_SIZE+2)? 1:0):0;
    assign spad_w_we_en = (encoder_state == load_wb)? ((cnt_wb < KERNEL_SIZE)? 1:0):0;
    
    assign encoder_w = (wbsz_init_en)? (((cnt_wb < KERNEL_SIZE) )?  sram_dout:0):0;
    always @(negedge sclk or negedge rst_n) begin
        if (!rst_n) begin
            scale <= 0;
            encoder_b <= 0;
        end
        else begin
            if (encoder_state == idle) begin
                scale <= 0;
                encoder_b <= 0;
            end
            else begin
                if (wbsz_init_en ) begin
                    encoder_b <= (cnt_wb == KERNEL_SIZE)? sram_dout:encoder_b;
                    scale <= (cnt_wb == KERNEL_SIZE+1)? sram_dout :scale;
                end
                else begin
                    encoder_b <= encoder_b;
                    scale <= scale;                    
                end
            end
            
        end

    end

    // assign encoder_b = (wbsz_init_en)? (((cnt_wb == KERNEL_SIZE))? sram_dout:encoder_b): encoder_b ;
    // assign scale     = (wbsz_init_en)? (((cnt_wb == KERNEL_SIZE+1))? sram_dout :scale): scale ;
    assign sram1_en = wbsz_init_en;
    
    always @(negedge sclk or negedge rst_n) begin
        if (!rst_n) begin
            addr_encoder_w_orgin <= 0;
            addr_encoder_b_orgin <= 0;
            addr_encoder_wb      <= 0; //address from sram
            spad_w_addr_we       <= 0;
            
        end
        else begin
            if (encoder_state == load_wb)  begin
                if (cnt_wb < (KERNEL_SIZE)) begin // store weight
                    if (cnt_wb == KERNEL_SIZE-1) begin
                        addr_encoder_w_orgin <= addr_encoder_wb + 1; //end of weight
                        addr_encoder_wb      <= addr_encoder_b_orgin;
                        spad_w_addr_we       <= 0; //??
                        addr_encoder_b_orgin <= addr_encoder_b_orgin;  //unchanged
                    end
                    else begin
                        addr_encoder_wb      <= addr_encoder_wb + 1'b1; //addr+1
                        spad_w_addr_we       <= spad_w_addr_we + 1; //spad addr + 1
                        addr_encoder_w_orgin <= addr_encoder_w_orgin; //unchanged
                        addr_encoder_b_orgin <= addr_encoder_b_orgin;   //unchANGED
                    end
                end
                else if (cnt_wb == (KERNEL_SIZE)) begin
                    addr_encoder_b_orgin <= addr_encoder_wb +1;
                    addr_encoder_wb      <= addr_encoder_output_scale;
                    addr_encoder_w_orgin <= addr_encoder_w_orgin; // unchanged
                    spad_w_addr_we       <= spad_w_addr_we; //unchanged

                end
                else if (cnt_wb == KERNEL_SIZE+1) begin
                    addr_encoder_w_orgin <= addr_encoder_w_orgin; //unchanged
                    spad_w_addr_we       <= spad_w_addr_we; //unchanged
                    addr_encoder_b_orgin <= addr_encoder_b_orgin; //unchanged
                    addr_encoder_wb <= addr_encoder_w_orgin;
                    
                end
            end
            else if (encoder_state == idle) begin
                addr_encoder_w_orgin <= addr_encoder_w_init;
                addr_encoder_b_orgin <= addr_encoder_b_init;
                addr_encoder_wb      <= addr_encoder_w_init; //address from sram
                spad_w_addr_we       <= 0;                
            end
            else ;
        end
    end
    // Outputs:load_a:spad_a_data_in
    //padding
    // assign shift_en       = (encoder_state == load_a)? 1: 0;

    // assign spad_a_data_in = (encoder_state == load_a)?input_signal[SPAD_DEPTH*DATA_DW-1:0]:spad_a_data_in;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            shift_en <= 0;
        end
        else begin
            if (encoder_state == load_a) begin
                if (cnt_lo == 0 ) shift_en <= 2;
                else if (cnt_lo == LENGTH_OUT-1) shift_en <= 3;
                else shift_en <= 1;
            end
            else begin
                shift_en <= 0;
            end
        end
    end
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            spad_a_data_in <= 0;
        end
            else begin
            if (encoder_state == load_a) begin
                if (cnt_lo == 0) spad_a_data_in <= {input_signal[(SPAD_DEPTH-PADDING_PRE)*DATA_DW-1:0],{PADDING_PRE*DATA_DW{1'b0}}};
                else if (cnt_lo == LENGTH_OUT-1) spad_a_data_in <= {{PADDING_POST*DATA_DW{1'b0}}, input_signal[(SPAD_DEPTH-PADDING_POST)*DATA_DW-1:0]};
                else spad_a_data_in <= input_signal[SPAD_DEPTH*DATA_DW-1:0];
            end
            else begin
                spad_a_data_in <= spad_a_data_in;
            end
        end
    end    

                    
                    
    // Outputs: mac: spad_w_addr_re, spad_a_addr_re
    always@(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            spad_w_addr_re <= 0 ;
            spad_a_addr_re <= 0;
        end
        else begin
            if (encoder_state == mac) begin
                spad_w_addr_re <= (spad_w_addr_re == KERNEL_SIZE-1)? spad_w_addr_re: spad_w_addr_re + 1;
                spad_a_addr_re <= (spad_a_addr_re == KERNEL_SIZE-1)? spad_a_addr_re:spad_a_addr_re + 1;
            end
            else begin
                spad_w_addr_re <= 0;
                spad_a_addr_re <= 0;
            end
        end
    end
    
    
    // Outputs: add_b:
    
    // Outputs: done:encoder_done
    assign encoder_done = (encoder_state == done) ? 1'b1 : 1'b0;
    
    // out_vld
    always@(posedge wclk or negedge rst_n)
    begin
        if (!rst_n) encoder_out_vld <= 0;
        else
        begin
            if (seg_state == 4'b0001) begin//encoder
                if (mult_a_crl == 2'b10) encoder_out_vld <= 1;
                else encoder_out_vld <= 0;
            end
            else begin
                encoder_out_vld <= 0;
            end

        end
    end        
    reg [1:0]  mult_a_crl_d;   
    always@(posedge wclk or negedge rst_n)
    begin
        if (!rst_n) mult_a_crl_d <= 0;
        else
        begin
            mult_a_crl_d <= mult_a_crl;

        end
    end     


    assign relu_en = mult_a_crl_d == 2'b11;
    assign round_en = mult_a_crl_d == 2'b10;

    // addr_encoder_sram_act, encoder_sram_act_din, encoder_sram_act_en, encoder_sram_act_we;
    reg encoder_sram_act_end;
    // wire encoder_out_vld_half;
    reg encoder_out_vld_d;
    always @(negedge sclk or negedge rst_n) begin
        if (!rst_n) begin
            addr_encoder_sram_act <= ADDR_ENCODER_SRAM_ACT_INIT;
            encoder_sram_act_end <= 0;
        end
        else begin
            if (encoder_out_vld_d) begin
                if (!encoder_sram_act_end) begin
                    addr_encoder_sram_act <= addr_encoder_sram_act + 1;
                    encoder_sram_act_end <= 1;
                end
                else begin
                    addr_encoder_sram_act <= addr_encoder_sram_act;
                    encoder_sram_act_end <= encoder_sram_act_end;              
                end
            end
            else if (encoder_state == idle) begin //rst
                addr_encoder_sram_act <= ADDR_ENCODER_SRAM_ACT_INIT;
                encoder_sram_act_end <= 0;             
            end
            else begin
                addr_encoder_sram_act <= addr_encoder_sram_act;
                encoder_sram_act_end <= 0;           
            end
        end
    end
    reg signed [DATA_OUT_DW-1:0] encoder_out_reg;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) encoder_out_reg <= 0;
        else encoder_out_reg <= encoder_out;
    end
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) encoder_out_vld_d <= 0;
        else encoder_out_vld_d <= encoder_out_vld;
    end    

    // assign encoder_out_vld_half = !wclk & encoder_out_vld;
    assign encoder_sram_act_din = (encoder_sram_act_we)? encoder_out_reg : 0;
    assign encoder_sram_act_en = encoder_out_vld_d;
    assign encoder_sram_act_we = (encoder_out_vld_d) & (!encoder_sram_act_end);

    // assign encoder_out_vld_half = !wclk & encoder_out_vld;
    // assign encoder_sram_act_din = (encoder_out_vld_half)? encoder_out:0;
    // assign encoder_sram_act_en = encoder_out_vld;
    // assign encoder_sram_act_we = (encoder_out_vld_half) & (!encoder_sram_act_end);

endmodule
                    
