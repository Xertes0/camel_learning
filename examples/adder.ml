open Camel_learning

let bit_count = 4

let n_bits_set n =
  Int.lognot (Int.shift_left (Int.shift_right (Int.lognot 0) n) n)

let (tx, ty) =
  let max = n_bits_set bit_count in

  let tx = Matrix.zeros ((max + 1) * (max + 1)) (bit_count * 2) in
  let ty = Matrix.zeros ((max + 1) * (max + 1)) (bit_count + 1) in

  let row = ref 0 in

  for a = 0 to max do
    for b = 0 to max do
      let res = a + b in
      for col = 0 to bit_count do
        let bit = Int.shift_left 1 col in
        Matrix.set ty !row (bit_count - col) (if (Int.logand res bit) > 0 then 1. else 0.);
        if col < bit_count then (
          Matrix.set tx !row col (if (Int.logand a bit) > 0 then 1. else 0.);
          Matrix.set tx !row (col + bit_count) (if (Int.logand b bit) > 0 then 1. else 0.);
        );
      done;
      row := !row + 1;
    done
  done;
  (tx, ty)

;;
Demo_sdl.run
  { tx = tx;
    ty = ty;
    shape = [| 8; 8; 5; |];
    rate = 1.;
    actf = Act_func.sigmoid;
  }
  ~goal:0.05
  ()
