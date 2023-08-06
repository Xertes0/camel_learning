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

let gradient_for_model (model: t): t =
  let ws = Array.init (Array.length model.ws)
             (fun i -> Mat.zeros model.ws.(i).rows model.ws.(i).cols) in
  let bs = Array.init (Array.length model.bs)
             (fun i -> Mat.zeros model.bs.(i).rows model.bs.(i).cols) in
  { ws = ws;
    bs = bs;
  }

let sigmoid (v: float): float =
  1. /. (1. +. Float.exp (-. v))

let forward (model: t) (input: Mat.t): Mat.t =
  let ws = Array.to_seq model.ws in
  let bs = Array.to_seq model.bs in
  let fwd_layer x w b =
    Mat.map sigmoid (Mat.sum (Mat.dot x w) b) in
  Seq.fold_left2 fwd_layer input ws bs

let cost (model: t) ~(tx: Mat.t) ~(ty: Mat.t): float =
  let sum = ref 0. in
  for t_row = 0 to tx.rows - 1 do
    let output = forward model (Mat.take_row tx t_row) in
    for y_col = 0 to ty.cols - 1 do
      let diff = (Mat.get ty t_row y_col) -.
                 (Mat.get output 0 y_col) in
      sum := !sum +. (diff ** 2.)
    done
  done;
  !sum

let finite_difference
      (model: t)
      ~(tx: Mat.t) ~(ty: Mat.t)
      ~(eps: float) ~(rate: float): t =
  let base_cost = cost model ~tx ~ty in
  let grad = gradient_for_model model in

  (* Weights *)
  for w_i = 0 to Array.length model.ws - 1 do
    for arr_i = 0 to Array.length model.ws.(w_i).arr - 1 do
      let old_val = model.ws.(w_i).arr.(arr_i) in
      model.ws.(w_i).arr.(arr_i) <-
        model.ws.(w_i).arr.(arr_i) +. eps;
      grad.ws.(w_i).arr.(arr_i) <-
        (((cost model ~tx ~ty) -. base_cost) /. eps) *. (-.rate);
      model.ws.(w_i).arr.(arr_i) <- old_val;
    done
  done;

  (* Biases *)
  for b_i = 0 to Array.length model.bs - 1 do
    for arr_i = 0 to Array.length model.bs.(b_i).arr - 1 do
      let old_val = model.bs.(b_i).arr.(arr_i) in
      model.bs.(b_i).arr.(arr_i) <-
        model.bs.(b_i).arr.(arr_i) +. eps;
      grad.bs.(b_i).arr.(arr_i) <-
        (((cost model ~tx ~ty) -. base_cost) /. eps) *. (-.rate);
      model.bs.(b_i).arr.(arr_i) <- old_val;
    done
  done;
  grad

let apply_gradient (model: t) (grad: t): unit =
  for w_i = 0 to Array.length model.ws - 1 do
    model.ws.(w_i) <-
      Mat.sum model.ws.(w_i) grad.ws.(w_i);
  done;
  for b_i = 0 to Array.length model.bs - 1 do
    model.bs.(b_i) <-
      Mat.sum model.bs.(b_i) grad.bs.(b_i);
  done

let evaluate (model: t) ~(tx: Mat.t) ~(ty: Mat.t): unit =
  for t_row = 0 to tx.rows - 1 do
    Printf.printf "=== Test %i/%i ===\n" t_row (tx.rows - 1);

    Printf.printf "For input:\n";
    let input = Mat.take_row tx t_row in
    Mat.show input;

    Printf.printf "Expected:\n";
    let expected = Mat.take_row ty t_row in
    Mat.show expected;

    Printf.printf "Got:\n";
    let output = forward model input in
    Mat.show output;

    let error_mat = Mat.map (fun x -> -. x) output
                    |> Mat.sum expected in
                    (* |> Mat.map (fun x -> x ** 2.) in *)
    let error = Array.fold_left ( +. ) 0. error_mat.arr in
    Printf.printf "Error: %f\n" error;
  done
