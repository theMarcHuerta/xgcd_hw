module xgcd_bitwise #(
    parameter int TOTAL_BITS   = 8,             // bit-width for a and b
    parameter int APPROX_BITS  = 4,             // number of bits used for approximate division
    parameter int COEFF_WIDTH  = TOTAL_BITS+1     // bit-width for coefficients x,y,u,v
)(
    input  logic                       clk,
    input  logic                       reset,   // synchronous active-high reset
    input  logic                       start,   // when high in IDLE state, latches inputs
    input  logic [TOTAL_BITS-1:0]      a_in,    // original input a
    input  logic [TOTAL_BITS-1:0]      b_in,    // original input b
    output logic                       done,    // asserted when computation finishes
    output logic [TOTAL_BITS-1:0]      gcd,     // final GCD output
    output logic signed [COEFF_WIDTH-1:0] x,    // coefficient such that: a_in*x + b_in*y = gcd
    output logic signed [COEFF_WIDTH-1:0] y,    // coefficient such that: a_in*x + b_in*y = gcd
);

  //-------------------------------------------------------------------------
  // Internal register declarations
  //-------------------------------------------------------------------------
  // a_reg and b_reg hold the current remainder values (assumed nonnegative).
  logic [TOTAL_BITS-1:0] a_reg, b_reg;
  // Coefficient registers (signed).
  logic signed [COEFF_WIDTH-1:0] x_reg, y_reg, u_reg, v_reg;

  // FSM states: IDLE, COMPUTE, DONE.
  typedef enum logic [1:0] {IDLE, COMPUTE, DONE} state_t;
  state_t state;

  //-------------------------------------------------------------------------
  // Helper functions
  //-------------------------------------------------------------------------
  // Compute the bit-length (position of MSB + 1) of an unsigned value.
  function automatic int bit_length(input logic [TOTAL_BITS-1:0] val);
    int i;
    begin
      bit_length = 0;
      for (i = TOTAL_BITS-1; i >= 0; i--) begin
        if (val[i]) begin
          bit_length = i + 1;
          break;
        end
      end
    end
  endfunction

  // Extract the top APPROX_BITS from x_val.
  // If the value has fewer than APPROX_BITS bits, shift left to pad.
  function automatic logic [APPROX_BITS-1:0] get_fixed_top_bits(input logic [TOTAL_BITS-1:0] x_val);
    int len;
    begin
      len = bit_length(x_val);
      if (len <= APPROX_BITS)
        get_fixed_top_bits = x_val << (APPROX_BITS - len);
      else
        get_fixed_top_bits = x_val >> (len - APPROX_BITS);
    end
  endfunction

  //-------------------------------------------------------------------------
  // Wire declarations for the LUT instance.
  // These wires carry the normalized top bits and the LUT output.
  //-------------------------------------------------------------------------
  wire [APPROX_BITS-1:0] a_top_wire;
  wire [APPROX_BITS-1:0] b_top_wire;
  // The LUT module produces a result that is APPROX_BITS bits wide.
  wire [APPROX_BITS-1:0] lut_quotient;

  // We use continuous assignments to compute the normalized top bits
  // from the current values of a_reg and b_reg.
  assign a_top_wire = get_fixed_top_bits(a_reg);
  assign b_top_wire = get_fixed_top_bits(b_reg);

  //-------------------------------------------------------------------------
  // Instantiate the LUT module.
  // This module precomputes: (a_top << APPROX_BITS) / b_top,
  // for normalized a_top and b_top (with leading 1 assumed).
  //-------------------------------------------------------------------------
  xgcd_lut #(
    .APPROX_BITS(APPROX_BITS)
  ) lut_inst (
    .a_top(a_top_wire),
    .b_top(b_top_wire),
    .lut_result(lut_quotient)
  );

  //-------------------------------------------------------------------------
  // Main FSM: One iteration per clock cycle.
  //-------------------------------------------------------------------------
  always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
      state           <= IDLE;
      done            <= 1'b0;
      a_reg           <= '0;
      b_reg           <= '0;
      x_reg           <= '0;
      y_reg           <= '0;
      u_reg           <= '0;
      v_reg           <= '0;
    end else begin
      case (state)
        IDLE: begin
          done <= 1'b0;
          if (start) begin
            x_reg <= 1;
            y_reg <= 0;
            u_reg <= 0;
            v_reg <= 1;
            // Ensure a_reg >= b_reg (swap if needed).
            if (a_in > b_in) begin
              a_reg <= a_in;
              b_reg <= b_in;
            end else begin
              a_reg <= b_in;
              b_reg <= a_in;
            end
            state <= COMPUTE;
          end
        end // IDLE

        COMPUTE: begin
          if (b_reg == 0) begin
            state <= DONE;
          end else begin
            logic [$clog2(TOTAL_BITS+1)-1:0] len_a, len_b;
            logic [$clog2(TOTAL_BITS+1)-1:0] shift_amount;
            // We'll use a local variable for the quotient.
            // Since the LUT returns only APPROX_BITS bits, we zero-extend it
            // to match the expected width (APPROX_BITS+TOTAL_BITS).
            logic [APPROX_BITS-1:0] quotient;
            // what we shift b, u, v by to reduce the multiplication
            logic [$clog2(TOTAL_BITS+1)-1:0] shift_vars_amount;
            // Extended coefficient widths to avoid overflow.
            logic signed [COEFF_WIDTH-1:0] b_shifted, u_shifted, v_shifted;
            logic signed [COEFF_WIDTH-1:0] b_mul, u_mul, v_mul;
            // logic for the double residual
            logic signed [TOTAL_BITS-1:0] residual, residual_two;
            bit was_negative, was_negative_two;

            logic signed [COEFF_WIDTH-1:0] new_x, new_y;
            logic [TOTAL_BITS-1:0] abs_residual, abs_residual_two;

            // Compute the bit lengths for a_reg and b_reg.
            len_a = bit_length(a_reg);
            len_b = bit_length(b_reg);
            shift_amount = len_a - len_b;

            // Instead of calling a function, we use our LUT instance's output.
            // Zero-extend the LUT result to match the width of 'quotient'.
            quotient = lut_quotient;

            // Default values for shifted signals.
            b_shifted  = b_reg;
            u_shifted  = u_reg;
            v_shifted  = v_reg;

            // If shift_amount is large, adjust b and coefficient signals.
            if (shift_amount > (APPROX_BITS - 1)) begin
              shift_vars_amount = shift_amount - (APPROX_BITS - 1);
              b_shifted  = b_reg << shift_vars_amount;
              u_shifted  = u_reg <<< shift_vars_amount; // arithmetic left shift
              v_shifted  = v_reg <<< shift_vars_amount;
              shift_amount = APPROX_BITS - 1;
            end

            // Compute Q = (quotient << shift_amount) >> APPROX_BITS.
            logic [2*APPROX_BITS-2:0] Q;
            Q = (quotient << shift_amount) >> APPROX_BITS;

            // Compute multiplications.
            b_mul = b_shifted * Q;
            u_mul = u_shifted * Q;
            v_mul = v_shifted * Q;

            // Compute residuals.
            residual = $signed(a_reg) - b_mul;
            residual_two = residual - $signed(b_reg);

            was_negative      = residual[WIDTH-1];
            abs_residual      = was_negative ? (~residual + 1) : residual;
            was_negative_two  = residual_two[WIDTH-1];
            abs_residual_two  = was_negative_two ? (~residual_two + 1) : residual_two;

            // Choose the residual with smaller magnitude.
            if (abs_residual_two < abs_residual) begin
              abs_residual = abs_residual_two;
              u_mul = u_mul + u_reg;
              v_mul = v_mul + v_reg;
              was_negative = was_negative_two;
            end

            // Update coefficients based on the sign of the residual.
            if (was_negative) begin
              new_x = -x_reg + u_mul;
              new_y = -y_reg + v_mul;
            end else begin
              new_x = x_reg - u_mul;
              new_y = y_reg - v_mul;
            end

            // Depending on the magnitude of the residual,
            // update (a_reg,x_reg,y_reg) or swap (a_reg,b_reg) and update coefficients.
            if (abs_residual > b_reg) begin
              a_reg <= abs_residual;
              x_reg <= new_x;
              y_reg <= new_y;
              // b_reg, u_reg, and v_reg remain unchanged.
            end else begin
              a_reg <= b_reg;
              b_reg <= abs_residual;
              x_reg <= u_reg;
              y_reg <= v_reg;
              u_reg <= new_x;
              v_reg <= new_y;
            end

          end
        end // COMPUTE

        DONE: begin
          done <= 1'b1;
          // Final assertion for Bézout's identity:
          assert ($signed(a_in)*x_reg + $signed(b_in)*y_reg == $signed(a_reg))
            else $error("Bézout identity not satisfied!");
          // Stay in DONE until reset.
        end

        default: state <= IDLE;
      endcase
    end
  end

  //-------------------------------------------------------------------------
  // Output assignments
  //-------------------------------------------------------------------------
  assign gcd = a_reg;
  assign x   = x_reg;
  assign y   = y_reg;

endmodule
