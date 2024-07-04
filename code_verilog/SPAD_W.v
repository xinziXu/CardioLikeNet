//SPAD_W
`timescale  1ns/100ps
module SPAD_W #(parameter WEIGHT_DW = 32,
                parameter DEPTH = 8)
               (
                input sclk,
                input									rst_n,                                                                                                       //reset, 0 is effective
                input signed [WEIGHT_DW -1 : 0] data_in,                                                                                   //读使能信号，高电平有�??
                input									we_en,
                input [$clog2(DEPTH)-1 : 0]				addr_we, 
                input [$clog2(DEPTH)-1 : 0]				addr_re, 
                output signed [WEIGHT_DW-1:0]			data_out);
    
    reg signed [WEIGHT_DW-1:0] mem [DEPTH-1 : 0];
    
    //  write data
    integer i;
    always @ (negedge sclk or negedge rst_n)
    begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i+1) begin
                mem[i] <= 0;
            end
        end
        else begin
            if (we_en) begin
                mem[addr_we] <= data_in;
            end
        end
    end
    // read data
    assign data_out = we_en ? 0 : mem[addr_re];
    
    
endmodule
