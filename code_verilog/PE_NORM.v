`timescale  1ns/100ps
module PE_NORM #(parameter DATA_DW = 8, 
                 OUT_BQ_DW = 32)
               (input wclk,
                 input rst_n,
                 input signed [DATA_DW-1: 0] spad_w_data,
                 input signed [DATA_DW-1: 0] spad_a_data,
                 input signed [DATA_DW-1: 0] hardmard_a,//hardmard multiplier
                 input signed [DATA_DW-1: 0] hardmard_b,//hardmard multiplier
                 input  [2:0] mult_int8_crl, // lstm-> 000:idle/reset, 001:gatess_mac, 011: transfer, 111:hold, 010: hardmard prod
                 input signed [OUT_BQ_DW-1: 0] psum_32b, // from adjacent pe
                 input [3:0] seg_state,
                 output signed [OUT_BQ_DW-1: 0] out_32b);


    //D:multiplier for 8bit*8bit
    reg signed [DATA_DW-1:0] mult_8a; //scale 
    reg signed [DATA_DW-1:0] mult_8b; //output
    wire signed [2*DATA_DW-1: 0] prod_ab; 
    assign  prod_ab =  mult_8a * mult_8b; 
    
    reg signed [OUT_BQ_DW-1: 0] out_temp_32b;
    assign out_32b = out_temp_32b;
    always@(posedge wclk or negedge rst_n)
    begin
        if (!rst_n)
        begin
            mult_8a   <= 0;
            mult_8b   <= 0;
            out_temp_32b <= 0;
        end
        else
        begin
            if (seg_state == 4'b0010) begin //lstm
                if (mult_int8_crl == 3'b001) begin //gate_mac
                    mult_8a   <= spad_w_data;
                    mult_8b   <= spad_a_data;
                    out_temp_32b <= out_temp_32b + prod_ab;
                end
                else if (mult_int8_crl == 3'b011) begin // transfer
                    mult_8a   <= 0;
                    mult_8b   <= 0;
                    out_temp_32b <= psum_32b;
                end
                else if (mult_int8_crl == 3'b111) begin// hold
                    mult_8a   <= mult_8a;
                    mult_8b   <= mult_8b;
                    out_temp_32b <= out_temp_32b;
                end
                else if (mult_int8_crl == 3'b010) begin //hard_mard_p
                    mult_8a   <= hardmard_a;
                    mult_8b   <= hardmard_b;
                    out_temp_32b <= prod_ab;
                end
                else begin //reset
                    mult_8a   <= 0;
                    mult_8b   <= 0;
                    out_temp_32b <= 0;
                end
            end
            else if (seg_state == 4'b0100) begin
                if (mult_int8_crl == 3'b001) begin //gate_mac
                    mult_8a   <= spad_w_data;
                    mult_8b   <= spad_a_data;
                    out_temp_32b <= out_temp_32b + prod_ab;
                end
                else if (mult_int8_crl == 3'b011) begin // transfer
                    mult_8a   <= 0;
                    mult_8b   <= 0;
                    out_temp_32b <= out_temp_32b + psum_32b;
                end
                else begin //reset
                    mult_8a   <= 0;
                    mult_8b   <= 0;
                    out_temp_32b <= 0;
                end                
            end 
            else begin //reset
                mult_8a   <= 0;
                mult_8b   <= 0;
                out_temp_32b <= 0;
            end           
        end
    end
    
endmodule
