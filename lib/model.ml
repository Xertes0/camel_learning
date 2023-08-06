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
