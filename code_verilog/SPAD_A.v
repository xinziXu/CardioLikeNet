//SPAD_A
`timescale  1ns/100ps
module SPAD_A #(parameter DATA_DW = 12,
                parameter DEPTH = 8
                )
               (
                input                                   sclk,
                input									rst_n	,   //低电平有效的复位信号
                input                                   is_sram_in,
                input  [DATA_DW -1 : 0]                 sram_data_in,
                input									we_en,
                input [$clog2(DEPTH)-1 : 0]				addr_we,                                      
                input [DATA_DW*DEPTH -1 : 0]            data_in,              
                input [$clog2(DEPTH)-1 : 0]				addr_re,
                output signed [DATA_DW-1:0]			data_out);
    
    reg signed [DATA_DW-1 : 0] mem [DEPTH-1 : 0];

    
    // write data
    // genvar i;
    // generate       
    // for (i = 0; i < DEPTH; i=i+1) begin: gen_spad_a
    // assign mem[i] = data_in[(i+1)*DATA_DW-1:i*DATA_DW];
    // end             
    // endgenerate
    integer i;
    integer j;
    always @ (negedge sclk or negedge rst_n)
    begin
        if (!rst_n) begin
            for (j = 0; j < DEPTH; j = j+1) begin
                mem[j] <= 0;
            end
        end
        else begin
            if (is_sram_in) begin
                if (we_en) begin
                    mem[addr_we] <= sram_data_in;
                end
            end
            else begin
                for (i = 0; i < DEPTH; i=i+1) begin
                    mem[i] <= data_in[(i+1)*DATA_DW-1-: DATA_DW];
                end
            end
        end
    end    
    
    // read data
    assign data_out = mem[addr_re];
    
endmodule
