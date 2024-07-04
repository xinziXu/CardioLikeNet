module PE_MAIN #(parameter WB_DW = 32,
                 A_DW = 12,
                 DATA_BQ_DW = 32,
                 ENCODER_SCALE_DW  = 32,
                 LSTM_SCALE_DW = 32,
                 ANN_WB_DW = 16,
                 FEATURE_SUM_DW = A_DW +4,
                 OUT_DW = 8)
               (input wclk,
                 input rst_n,
                 input signed [WB_DW-1: 0] spad_w_data, //encoder
                 input signed [A_DW-1: 0] spad_a_data, //encoder
                 input signed [WB_DW-1: 0] encoder_b, //encoder
                 input signed [ENCODER_SCALE_DW -1 : 0] encoder_scale, 
                 input signed [WB_DW-1: 0] lstm_b,
                 input signed [2*(2*OUT_DW+LSTM_SCALE_DW)-1: 0] lstm_ct_temp_in_cat,//gates_scale * f_t[hs] * c_t[hs], from ct_buffer
                 input signed [DATA_BQ_DW-1:0] out_bq, // lstm: from each PEs
                 input signed [LSTM_SCALE_DW -1 : 0] scale,
                 input signed [DATA_BQ_DW-1:0] out_bq2, // lstm: from each PEs
                 input signed [LSTM_SCALE_DW -1 : 0] scale2,
                 input signed [WB_DW-1:0]decoder_b1,
                 input signed [WB_DW-1:0]decoder_b2,
                 input signed [DATA_BQ_DW-1:0] dcnn1_temp_value_for_1,
                 input signed [DATA_BQ_DW-1:0] psum_32b_8,
                 input signed [DATA_BQ_DW-1:0] psum_32b_16,
                 input signed [DATA_BQ_DW-1:0] psum_32b_24,
                 input signed [DATA_BQ_DW-1:0] psum_32b_32,
                 input signed [DATA_BQ_DW-1:0] psum_32b_32_d,
                 input [3:0]  seg_state, //0000:idle, 0001:encoder, 0010:lstm
                 input [2:0] decoder_top_state,
                 input cnn22_is_first,
                 input [1:0] mult_a_crl,  // encoder: 00: idle, 01: mac, 11: add_b, 10: requantization; lstm->00:idle, 01:requantization gates, 11 requan tail  
                 input [1:0] mult_b_crl, // encoder:  00: idle, 01: mac, 11: add_b, 10: requantization; lstm->00:idle, 01:requantization gates, 11 requan tail
                 input  [2:0] mult_int8_crl, //lstm-> 000:idle/reset, 001:gatess_mac, 011: transfer, 111:hold, 010: hardmard prod
                 input [1:0] add_a_crl, //lstm-> 00:idle,  01: add gates, 11: add_tail
                 input [1:0] add_b_crl, //lstm-> 00:idle,  01: add gates, 11: add_tail
                 input mult_out_round_en,
                 input sum_a_final_en,
                 input sum_b_final_en,
                 input encoder_relu_en,
                 input encoder_round_en,


                 input signed [OUT_DW -1 : 0] hardmard_a,
                 input signed [OUT_DW -1 : 0] hardmard_b,
                 input signed [DATA_BQ_DW-1: 0] psum_32b, //from adjacent pes
                 output signed [OUT_DW-1: 0] encoder_out,
                 output signed [OUT_DW-1:0] sum_a_final,
                 output signed [OUT_DW-1:0] sum_b_final,
                //  output signed [OUT_DW-1:0] out_temp_A_final,
                 output signed [2*OUT_DW+LSTM_SCALE_DW-1:0] lstm_hardmard_temp_a,
                 output signed [2*OUT_DW+LSTM_SCALE_DW-1:0] lstm_hardmard_temp_b,
                 output signed [OUT_DW-1:0] mult_a_out_round,
                 output signed [OUT_DW-1:0] mult_b_out_round,
                 output signed [DATA_BQ_DW-1: 0] out_32b,//to adjacent pes
                 input [3:0] top_state,
                 input signed [ANN_WB_DW-1:0] ann_b,
                 input [4:0] ann_state ,
                 input ann_mi_1,
                 input ann_mi_2,
                 input signed [ANN_WB_DW +A_DW  -1:0] ann_hidden_in,
                 input signed [FEATURE_SUM_DW  -1:0] ann_mi_in,
                 input signed [FEATURE_SUM_DW + ANN_WB_DW - 1:0] ann_mi_hidden_in,
                 output signed [ANN_WB_DW +A_DW  -1:0] ann_out,
                 output signed [ANN_WB_DW + FEATURE_SUM_DW -1:0] ann_out_mi,
                 input ann_relu_en); 
    localparam OUT_BQ_DW_ENCODER = A_DW + WB_DW;
    
    localparam dcnn1 =  3'b001; //from sram
    localparam cnn11 = 3'b011;
    localparam cnn12 = 3'b111;
    localparam dcnn2 = 3'b101;
    localparam cnn21  = 3'b110 ;
    localparam cnn22 = 3'b010;
    //A: multiplier for the first layer mac and lstm requantization, width is set to the same as requantization
    reg signed [LSTM_SCALE_DW-1:0] mult_w;  
    reg signed [DATA_BQ_DW-1:0] mult_a;
    wire signed [LSTM_SCALE_DW + DATA_BQ_DW - 1 : 0] prod;
    assign  prod = mult_w * mult_a;

    localparam HARDMARD_PROD_DW = 2*OUT_DW;
    assign lstm_hardmard_temp_a = prod[HARDMARD_PROD_DW+LSTM_SCALE_DW-1:0]; //for the requantization of lstm  hardmard product,
    
    
    // need to choose
    wire round_carry_out_mult_a;
    wire signed [DATA_BQ_DW+1:0] mult_a_temp;    
    assign round_carry_out_mult_a =( mult_out_round_en)? ((prod[LSTM_SCALE_DW+DATA_BQ_DW-1])? (prod[LSTM_SCALE_DW - 2]&(|prod[LSTM_SCALE_DW - 3:0])) : prod[LSTM_SCALE_DW - 2]):0;
    assign mult_a_temp =( mult_out_round_en)?( {prod[LSTM_SCALE_DW+DATA_BQ_DW-1],prod[LSTM_SCALE_DW+DATA_BQ_DW-1:LSTM_SCALE_DW-1]} + round_carry_out_mult_a):0 ;

    assign mult_a_out_round = ( mult_out_round_en)?((mult_a_temp>127)? 127:((mult_a_temp <-127)?-127 :mult_a_temp)):0;// avoid overlow




    //B: multipler for requantization
    reg signed [LSTM_SCALE_DW-1:0] mult_s; //scale 
    reg signed [OUT_BQ_DW_ENCODER - 1 : 0] mult_o; //output
    wire signed [LSTM_SCALE_DW + OUT_BQ_DW_ENCODER  -1 : 0] prod_so;
    assign prod_so     = mult_s * mult_o;
    assign lstm_hardmard_temp_b = prod_so[HARDMARD_PROD_DW+LSTM_SCALE_DW-1:0];

    // need to choose
    wire round_carry_out_mult_b;
    wire signed [DATA_BQ_DW+1:0] mult_b_temp;    
    assign round_carry_out_mult_b = ( mult_out_round_en)?( (prod_so[LSTM_SCALE_DW+DATA_BQ_DW-1])? (prod_so[LSTM_SCALE_DW - 2]&(|prod_so[LSTM_SCALE_DW - 3:0])) : prod_so[LSTM_SCALE_DW - 2]):0;
    assign mult_b_temp = ( mult_out_round_en)?({prod_so[LSTM_SCALE_DW+DATA_BQ_DW-1],prod_so[LSTM_SCALE_DW+DATA_BQ_DW-1:LSTM_SCALE_DW-1]} + round_carry_out_mult_b):0 ;
    assign mult_b_out_round =( mult_out_round_en)?( (mult_b_temp>127)? 127:((mult_b_temp <-127)?-127 :mult_b_temp)):0;// avoid overlow

    // adder_a
    reg signed [LSTM_SCALE_DW+DATA_BQ_DW -1:0]  add_a1; // the width may need to change
    reg signed [LSTM_SCALE_DW+DATA_BQ_DW-1:0]  add_a2;
    wire signed [LSTM_SCALE_DW+DATA_BQ_DW:0] sum_a;
    assign sum_a = add_a1 + add_a2;

    // need to choose
    wire round_carry_out_sum_a;
    wire signed [DATA_BQ_DW+1:0] sum_a_temp;    
    assign round_carry_out_sum_a = (sum_a_final_en)?( (sum_a[LSTM_SCALE_DW+DATA_BQ_DW])? (sum_a[LSTM_SCALE_DW - 2]&(|sum_a[LSTM_SCALE_DW - 3:0])) : sum_a[LSTM_SCALE_DW - 2]) :0;
    assign sum_a_temp =  (sum_a_final_en)?({sum_a[LSTM_SCALE_DW+DATA_BQ_DW],sum_a[LSTM_SCALE_DW+DATA_BQ_DW:LSTM_SCALE_DW-1]} + round_carry_out_sum_a):0 ;

    assign sum_a_final =  (sum_a_final_en)? ((sum_a_temp>127)? 127:((sum_a_temp <-127)?-127 :sum_a_temp)):0;// avoid overlow


    // adder_b
    reg signed [LSTM_SCALE_DW+DATA_BQ_DW:0]  add_b1;
    reg signed [LSTM_SCALE_DW+DATA_BQ_DW:0]  add_b2;
    wire signed [LSTM_SCALE_DW+DATA_BQ_DW+1:0] sum_b;
    assign sum_b = add_b1 + add_b2;

    // need to choose
    wire round_carry_out_sum_b;
    wire signed [DATA_BQ_DW+2:0] sum_b_temp;    
    assign round_carry_out_sum_b =  (sum_b_final_en)? ((sum_b[LSTM_SCALE_DW+DATA_BQ_DW+1])? (sum_b[LSTM_SCALE_DW - 2]&(|sum_b[LSTM_SCALE_DW - 3:0])) : sum_b[LSTM_SCALE_DW - 2]):0;
    assign sum_b_temp =(sum_b_final_en)? ( {sum_b[LSTM_SCALE_DW+DATA_BQ_DW+1],sum_b[LSTM_SCALE_DW+DATA_BQ_DW+1:LSTM_SCALE_DW-1]} + round_carry_out_sum_b):0 ;
    assign sum_b_final =(sum_b_final_en)?( (sum_b_temp>127)? 127:((sum_b_temp <-127)?-127 :sum_b_temp)):0; // avoid overlow

    //first layer 
    reg signed [ENCODER_SCALE_DW + DATA_BQ_DW -1:0] out_temp_A; //
    wire signed [A_DW + WB_DW -1:0] encoder_out_bq; //the first layer, results from 8 macs and bias adding (the first layer)
    assign encoder_out_bq = (encoder_relu_en)?((out_temp_A > 0)? out_temp_A : 0):0;

    // need to choose
    localparam encoder_shift = WB_DW-1+12; //for the quantization of the encoder out, 12 is the decimal bit for encoder scale
    wire signed [ OUT_BQ_DW_ENCODER-1:0] encoder_out_temp ; 
    wire round_carry;
    assign round_carry = (encoder_round_en)? ((prod_so[ENCODER_SCALE_DW+DATA_BQ_DW-1])? (prod_so[encoder_shift - 1]&(|prod_so[encoder_shift - 2:0])) : prod_so[encoder_shift - 1]):0; // carry for round
    assign encoder_out_temp = (encoder_round_en)? ({prod_so[ENCODER_SCALE_DW+DATA_BQ_DW-1], prod_so[ENCODER_SCALE_DW+DATA_BQ_DW-1:encoder_shift]} + round_carry) :0;
    assign encoder_out       = (encoder_round_en)? ((encoder_out_temp> 127)? 127: ((encoder_out_temp<-127)?-127:encoder_out_temp)):0;   //torch.clamp(torch.round(cnn_out/encoder_out_scale/(2**31)),-127,127)



// ann layer

    localparam  ANN_SHIFT = 10;
    assign ann_out = (ann_relu_en &( (ann_state == 5'd1)|(ann_state == 5'd2)))?((out_temp_A > 0)? out_temp_A : 0):0;
    assign ann_out_mi = (ann_relu_en & (ann_mi_1|ann_mi_2))?((out_temp_A > 0)? out_temp_A : 0):0;
    

    //C:multiplier for 8bit*8bit
    reg signed [OUT_DW-1:0] mult_8a; //scale 
    reg signed [OUT_DW-1:0] mult_8b; //output
    wire signed [2*OUT_DW-1: 0] prod_ab; 
    assign  prod_ab =  mult_8a * mult_8b;
    
    
    always@(posedge wclk or negedge rst_n)
    begin
        if (!rst_n)
        begin
            mult_w   <= 0;
            mult_a   <= 0;
            out_temp_A <= 0;
        end
        else
        begin
            if (top_state == 4'b0010) begin 
                if (seg_state == 4'b0001) begin //encoder
                    if (mult_a_crl == 2'b01) begin
                        mult_w     <= spad_w_data;
                        mult_a     <= spad_a_data;
                        out_temp_A <= out_temp_A + prod;
                    end
                    else if (mult_a_crl == 2'b11) begin
                        mult_w     <= mult_w;
                        mult_a     <= mult_a;
                        out_temp_A <= out_temp_A + encoder_b;
                    end
                    else if (mult_a_crl == 2'b10) begin
                        mult_w   <= mult_w;
                        mult_a   <= mult_a;
                        out_temp_A <= out_temp_A;
                    end
                    else begin
                        mult_w   <= 0;
                        mult_a   <= 0;
                        out_temp_A <= 0;
                    end
                end
                else if (seg_state == 4'b0010) begin //lstm
                    if ((mult_a_crl == 2'b01) | (mult_a_crl == 2'b11)) begin
                        mult_w   <= scale;
                        mult_a   <= out_bq;
                        out_temp_A <= prod;                    
                    end
                    else begin
                        mult_w   <= 0;
                        mult_a   <= 0;
                        out_temp_A <= 0;                    
                    end
                end
                else if (seg_state == 4'b0100) begin
                    if (decoder_top_state== dcnn1) begin
                        if (mult_a_crl == 2'b10) begin
                            mult_w   <= scale;
                            mult_a   <= psum_32b_32; // ????
                            out_temp_A <= prod;                    
                        end
                        else begin
                            mult_w   <= 0;
                            mult_a   <= 0;
                            out_temp_A <= 0;                    
                        end 
                    end 
                    else if (decoder_top_state== dcnn2)     begin
                        if (mult_a_crl == 2'b10) begin
                            mult_w   <= scale;
                            mult_a   <= psum_32b_16; // ????
                            out_temp_A <= prod;                    
                        end
                        else begin
                            mult_w   <= 0;
                            mult_a   <= 0;
                            out_temp_A <= 0;                    
                        end                     
                    end      
                    else if ((decoder_top_state==cnn11)|(decoder_top_state==cnn12)|(decoder_top_state==cnn21)|(decoder_top_state==cnn22)) begin
                        if (mult_a_crl == 2'b10) begin
                            mult_w   <= scale;
                            mult_a   <= sum_a; // ????
                            out_temp_A <= prod;                    
                        end
                        else begin
                            mult_w   <= 0;
                            mult_a   <= 0;
                            out_temp_A <= 0;                    
                        end                       
                    end
                    else begin
                        mult_w   <= 0;
                        mult_a   <= 0;
                        out_temp_A <= 0;   
                    end   
                end
            end
            else if (top_state == 4'b0101) begin //ann
                if (ann_state == 5'd1) begin
                    if (mult_a_crl == 2'b01) begin
                        mult_w     <= spad_w_data;
                        mult_a     <= spad_a_data;
                        out_temp_A <= out_temp_A + prod;
                    end
                    else if (mult_a_crl == 2'b11) begin
                        mult_w     <= mult_w;
                        mult_a     <= mult_a;
                        out_temp_A <= out_temp_A + ann_b;
                    end 
                    else if (mult_a_crl == 2'b10) begin
                        mult_w   <= 0;
                        mult_a   <= 0;
                        out_temp_A <= out_temp_A;
                    end
                    else begin
                        mult_w   <= 0;
                        mult_a   <= 0;
                        out_temp_A <= 0;
                    end          
                end
                else if (ann_state ==  5'd2)      begin
                    if (mult_a_crl == 2'b01) begin
                        mult_w     <= spad_w_data;
                        mult_a     <= ann_hidden_in;
                        out_temp_A <= out_temp_A + prod;
                    end
                    else if (mult_a_crl == 2'b11) begin
                        mult_w     <= mult_w;
                        mult_a     <= mult_a;
                        out_temp_A <= (out_temp_A>>>ANN_SHIFT) + ann_b;
                    end  
                    else if (mult_a_crl == 2'b10) begin
                        mult_w   <= 0;
                        mult_a   <= 0;
                        out_temp_A <= out_temp_A;
                    end
                    else begin
                        mult_w   <= 0;
                        mult_a   <= 0;
                        out_temp_A <= 0;
                    end                    
                end
                else if (ann_mi_1)      begin
                    if (mult_a_crl == 2'b01) begin
                        mult_w     <= spad_w_data;
                        mult_a     <= ann_mi_in;
                        out_temp_A <= out_temp_A + prod;
                    end
                    else if (mult_a_crl == 2'b11) begin
                        mult_w     <= mult_w;
                        mult_a     <= mult_a;
                        out_temp_A <= out_temp_A + ann_b;
                    end  
                    else if (mult_a_crl == 2'b10) begin
                        mult_w   <= 0;
                        mult_a   <= 0;
                        out_temp_A <= out_temp_A;
                    end
                    else begin
                        mult_w   <= 0;
                        mult_a   <= 0;
                        out_temp_A <= 0;
                    end                    
                end 
                else if (ann_mi_2)      begin
                    if (mult_a_crl == 2'b01) begin
                        mult_w     <= spad_w_data;
                        mult_a     <= ann_mi_hidden_in;
                        out_temp_A <= out_temp_A + prod;
                    end
                    else if (mult_a_crl == 2'b11) begin
                        mult_w     <= mult_w;
                        mult_a     <= mult_a;
                        out_temp_A <=  (out_temp_A>>>ANN_SHIFT)  + ann_b;
                    end  
                    else if (mult_a_crl == 2'b10) begin
                        mult_w   <= 0;
                        mult_a   <= 0;
                        out_temp_A <= out_temp_A;
                    end
                    else begin
                        mult_w   <= 0;
                        mult_a   <= 0;
                        out_temp_A <= 0;
                    end                    
                end      
            end
            else if (top_state == 4'b1111) begin
                mult_w   <= 0;
                mult_a   <= 0;
                out_temp_A <= 0;                
            end
        end
    end


    always@(posedge wclk or negedge rst_n)
    begin
        if (!rst_n)
        begin
            mult_s  <= 0;
            mult_o  <= 0;
        end
        else
        begin
            if (seg_state == 4'b0001) begin
                if (mult_b_crl == 2'b10) begin
                    mult_s  <= encoder_scale;
                    mult_o  <= encoder_out_bq;
                end
                else begin
                    mult_s  <= mult_s;
                    mult_o  <= mult_o;
                end
            end
            else if (seg_state == 4'b0010) begin //lstm
                if ((mult_b_crl == 2'b01) | (mult_b_crl == 2'b11)) begin
                    mult_s  <= scale2;
                    mult_o  <= out_bq2;
                end
                else begin
                    mult_s  <= 0;
                    mult_o  <= 0;
                end                
            end
            else if (seg_state == 4'b0100) begin
                if (decoder_top_state== dcnn2)     begin
                    if (mult_b_crl == 2'b10) begin
                        mult_s   <= scale;
                        mult_o   <= psum_32b_32; // ????                 
                    end
                    else begin
                        mult_s   <= 0;
                        mult_o   <= 0;              
                    end                     
                end      
                else if ((decoder_top_state==cnn21) |(decoder_top_state==cnn22)) begin
                    if (mult_b_crl == 2'b10) begin

                        mult_s   <= scale;
                        mult_o   <= sum_b; // ????
               
                    end
                    else begin
                        mult_s   <= 0;
                        mult_o   <= 0;                
                    end                     
                end   
            end
        end
    end



    reg signed [DATA_BQ_DW-1: 0] out_temp_32b;
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
                    out_temp_32b <= out_temp_32b + dcnn1_temp_value_for_1;
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
    localparam HARDMARD_PROD_OUT_DW = HARDMARD_PROD_DW+LSTM_SCALE_DW;
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            add_a1 <= 0;
            add_a2 <= 0;
        end
        else begin
            if (seg_state == 4'b0010) begin
                if(add_a_crl == 2'b01) begin
                    add_a1 <= prod;
                    add_a2 <= prod_so;
                end
                else if (add_a_crl == 2'b11) begin
                    add_a1 <= {{(LSTM_SCALE_DW+DATA_BQ_DW-HARDMARD_PROD_OUT_DW ){lstm_ct_temp_in_cat[HARDMARD_PROD_OUT_DW-1]}},lstm_ct_temp_in_cat[HARDMARD_PROD_OUT_DW-1-:HARDMARD_PROD_OUT_DW]};
                    add_a2 <= prod;    //       
                end
                else begin
                    add_a1 <= 0;
                    add_a2 <= 0;                
                end
            end
            else if (seg_state == 4'b0100) begin
                
                // if(add_a_crl == 2'b10) begin
                //     add_a1 <= prod;
                //     add_a2 <= decoder_b1;
                // end       
                // else begin
                //     add_a1 <= 0;
                //     add_a2 <= 0;                    
                // end
                if ((decoder_top_state == cnn11) |(decoder_top_state == cnn12)) begin
                    if(add_a_crl == 2'b10) begin
                        add_a1 <= psum_32b_32;
                        add_a2 <= decoder_b1;                        
                    end
                    else begin
                        add_a1 <= 0;
                        add_a2 <= 0;                          
                    end
                end   
                else if ((decoder_top_state == cnn21)|(decoder_top_state == cnn22)) begin
                    if(add_a_crl == 2'b10) begin
                        if (cnn22_is_first) begin
                            add_a1 <= psum_32b_8;
                            add_a2 <= decoder_b1;                               
                        end
                        else begin
                            add_a1 <= psum_32b_24;
                            add_a2 <= decoder_b1;                            
                        end
                     
                    end
                    else begin
                        add_a1 <= 0;
                        add_a2 <= 0;                          
                    end                    
                end
                else begin
                    add_a1 <= 0;
                    add_a2 <= 0;                      
                end

            end
        end        
    end
    always @(posedge wclk or negedge rst_n) begin
        if (!rst_n) begin
            add_b1 <= 0;
            add_b2 <= 0;
        end
        else begin
            if (seg_state == 4'b0010) begin
                if(add_b_crl == 2'b01) begin
                    add_b1 <= sum_a;
                    add_b2 <= lstm_b<<<7;
                end
                else if (add_b_crl == 2'b11) begin
                    add_b1 <= {{(LSTM_SCALE_DW+DATA_BQ_DW+1-HARDMARD_PROD_OUT_DW){lstm_ct_temp_in_cat[2*HARDMARD_PROD_OUT_DW-1]}},lstm_ct_temp_in_cat[2*HARDMARD_PROD_OUT_DW-1-:HARDMARD_PROD_OUT_DW]};
                    add_b2 <= prod_so;//               
                end
                else begin
                    add_b1 <= 0;
                    add_b2 <= 0;                
                end
            end
            else if (seg_state == 4'b0100) begin   
                if ((decoder_top_state == cnn21)|(decoder_top_state == cnn22)) begin
                    if (add_b_crl == 2'b10) begin
                        if( cnn22_is_first) begin
                            add_b1 <= psum_32b_16;
                            add_b2 <= decoder_b2;
                        end 
                        else begin
                            add_b1 <= psum_32b_32_d;
                            add_b2 <= decoder_b2;                            
                        end
                    end
                    else begin
                        add_b1 <= 0;
                        add_b2 <= 0;
                    end
                end  
                else begin
                    add_b1 <= 0;
                    add_b2 <= 0;                    
                end          
                // if(add_b_crl == 2'b10) begin
                //     add_b1 <= prod_so;
                //     add_b2 <= decoder_b2;
                // end       
                // else begin
                //     add_b1 <= 0;
                //     add_b2 <= 0;                    
                // end                   
            end
        end        
    end

    



endmodule
