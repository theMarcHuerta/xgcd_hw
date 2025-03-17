module xgcd_lut #(
    parameter int APPROX_BITS = 3,  // number of bits (e.g. 3: valid values 100 to 111)
    // Address width: combine the (APPROX_BITS-1) lower bits of each normalized input.
    localparam int LUT_ADDR_WIDTH = 2*APPROX_BITS - 2,
    localparam int LUT_DEPTH = 1 << LUT_ADDR_WIDTH,
    // The LUT result is fixed-point with APPROX_BITS bits.
    localparam int LUT_WIDTH = APPROX_BITS
)(
    input  logic [APPROX_BITS-1:0] a_top,  // normalized, MSB is always 1
    input  logic [APPROX_BITS-1:0] b_top,  // normalized, MSB is always 1
    output logic [LUT_WIDTH-1:0]   lut_result
);

  //-------------------------------------------------------------------------
  // Declare a one-dimensional LUT ROM with LUT_DEPTH entries.
  // Each entry is LUT_WIDTH bits wide.
  //-------------------------------------------------------------------------
  logic [LUT_WIDTH-1:0] lut_table [0:LUT_DEPTH-1];

  //-------------------------------------------------------------------------
  // Generate the LUT entries.
  // Since a_top and b_top are assumed normalized (leading 1), their valid range is:
  //   from (1 << (APPROX_BITS-1)) to ( (1 << APPROX_BITS) - 1 ).
  // The effective variable part is the lower APPROX_BITS-1 bits.
  // We map a_top and b_top into a unique index as:
  //     index = { a_top[APPROX_BITS-2:0], b_top[APPROX_BITS-2:0] }
  // For each valid combination, precompute:
  //     result = (a_top << APPROX_BITS) / b_top,
  // which is expected to fit in LUT_WIDTH bits.
  //-------------------------------------------------------------------------
  genvar i, j;
  generate
    // Loop over valid values for a_top.
    for (i = (1 << (APPROX_BITS-1)); i < (1 << APPROX_BITS); i = i + 1) begin : gen_a
      // Loop over valid values for b_top.
      for (j = (1 << (APPROX_BITS-1)); j < (1 << APPROX_BITS); j = j + 1) begin : gen_b
        // Compute a unique index based on the variable (lower) bits:
        localparam int index = ((i - (1 << (APPROX_BITS-1))) << (APPROX_BITS-1))
                                 | (j - (1 << (APPROX_BITS-1)));
        // Precompute the fixed-point division result.
        // Here (i << APPROX_BITS) shifts i by APPROX_BITS, then divides by j.
        localparam logic [LUT_WIDTH-1:0] precomp_val = (i << APPROX_BITS) / j;
        // Initialize the LUT entry.
        initial begin
          lut_table[index] = precomp_val;
        end
      end
    end
  endgenerate

  //-------------------------------------------------------------------------
  // Combinational logic to form the LUT address and return the precomputed value.
  //-------------------------------------------------------------------------
  always_comb begin
    // Extract the variable (lower) bits from each normalized input.
    // Because the MSB is always 1, we can ignore it in the address.
    logic [LUT_ADDR_WIDTH-1:0] addr;
    addr = { a_top[APPROX_BITS-2:0], b_top[APPROX_BITS-2:0] };
    lut_result = lut_table[addr];
  end

endmodule
