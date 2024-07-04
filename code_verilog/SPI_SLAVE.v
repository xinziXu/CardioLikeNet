module SPI_SLAVE #(parameter SPI_DW = 16
)(
    input wclk,
    input rst_n,
    input spi_clk,
    input cs_n,
    input mosi,
    input [SPI_DW-1:0] spi_din,
    output reg dout_vld,
    output      miso,
    output reg [SPI_DW-1:0] spi_dout
);
    
reg [$clog2(SPI_DW+1)-1:0] bit_cnt;
reg [SPI_DW-1:0] data;
// wire spi_clk_fan;
// assign spi_clk_fan = !spi_clk;
always @(posedge wclk or negedge rst_n) begin
    if (!rst_n) begin
        dout_vld <= 0;
        bit_cnt <= 0;
        spi_dout <= 0;
        data <= 0;
        dout_vld <= 1'b0;

    end
    else begin
        
        if (cs_n) begin
            data <= spi_din;
            bit_cnt <= 0;
            dout_vld <= 1'b0; 
        end
        else begin
            if ( spi_clk) begin //falling edge
                data <= {data[SPI_DW-2:0], mosi};
                bit_cnt <= bit_cnt + 1'b1;     
                if (bit_cnt == SPI_DW ) begin
                    dout_vld <= 1'b1; 
                    spi_dout <= data;
                    data <= spi_din;
                end
                else begin
                    dout_vld <= 1'b0;
                end
            end
            else begin
                dout_vld <= 1'b0;
            end 
        end
    end
end
assign miso = data[SPI_DW-1];
endmodule