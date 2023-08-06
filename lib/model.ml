module Mat = Matrix

type t =
  { ws : Matrix.t array;
    bs : Matrix.t array;
  }

let random (shape: int array): t =
  let ws = Array.init (Array.length shape - 1)
             (fun i -> Mat.random shape.(i) shape.(i+1)) in
  let bs = Array.init (Array.length shape - 1)
             (fun i -> Mat.random 1 shape.(i+1)) in
  { ws = ws;
    bs = bs;
  }

let forward (model: t) (input: Mat.t): Mat.t =
  let ws = Array.to_seq model.ws in
  let bs = Array.to_seq model.bs in
  let fwd_layer x w b = Mat.sum (Mat.dot x w) b in
  Seq.fold_left2 fwd_layer input ws bs

let cost (model: t) (tx: Mat.t) (ty: Mat.t): float =
  let sum = ref 0. in
  for t_row = 0 to tx.rows - 1 do
    let output = forward model (Mat.take_row tx t_row) in
    for ty_col = 0 to ty.cols - 1 do
      let diff = (Mat.get ty t_row ty_col) -.
                 (Mat.get output 0 ty_col) in
      sum := !sum +. (diff *. diff)
    done
  done;
  !sum

(* OR gate *)
let demo () =
  let tx = Mat.from_3d_list
                  [ [ 0.; 0. ];
                    [ 0.; 1. ];
                    [ 1.; 0. ];
                    [ 1.; 1. ] ] in
  let ty = Mat.from_3d_list
                  [ [ 0. ];
                    [ 1. ];
                    [ 1. ];
                    [ 1. ] ] in
  let shape = [| 2; 2; 1; |] in
  let model = random shape in
  cost model tx ty
