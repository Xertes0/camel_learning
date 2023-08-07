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

(*
  Like forward but returns an array where:
   - first item is the given input
   - last item is the model's output
   - every item in between is the result of
     passing the previous item by one layer
 *)
let forward_collect (model: t) (input: Mat.t): Mat.t array =
  let ws = Array.to_seq model.ws in
  let bs = Array.to_seq model.bs in
  let fwd_layer x (w, b) =
    Mat.map sigmoid (Mat.sum (Mat.dot x w) b) in
  Seq.scan fwd_layer input (Seq.zip ws bs)
  |> Array.of_seq

let back_propagation
      (model: t)
      ~(tx: Mat.t) ~(ty: Mat.t)
      ~(rate: float): t =
  let grad = gradient_for_model model in
  for t_row = 0 to tx.rows - 1 do
    let tx_row = Mat.take_row tx t_row in
    let model_a = forward_collect model tx_row in
    let grad_a = Array.init (Array.length model_a) (fun i -> Mat.zeros model_a.(i).rows model_a.(i).cols) in
    let a_len = Array.length model_a in
    for t_col = 0 to ty.cols - 1 do
      Mat.set grad_a.(a_len - 1) 0 t_col
        ((Mat.get model_a.(a_len - 1) 0 t_col) -.
           (Mat.get ty t_row t_col));
    done;
    for a_i = a_len - 1 downto 1 do
      for a_col = 0 to grad_a.(a_i).cols - 1 do
        let sigm = Mat.get model_a.(a_i) 0 a_col in
        let first = 2. *. (Mat.get grad_a.(a_i) 0 a_col) *. sigm *. (1. -. sigm) in
        Mat.set grad.bs.(a_i - 1) 0 a_col
          ((Mat.get grad.bs.(a_i - 1) 0 a_col) +. first);
        for w_row = 0 to grad.ws.(a_i - 1).rows - 1 do
          Mat.set grad.ws.(a_i - 1) w_row a_col
            ((Mat.get grad.ws.(a_i - 1) w_row a_col) +.
               (first *. (Mat.get model_a.(a_i - 1) 0 w_row)));
          Mat.set grad_a.(a_i - 1) 0 w_row
            ((Mat.get grad_a.(a_i - 1) 0 w_row) +.
               (first *. (Mat.get model.ws.(a_i - 1) w_row a_col)));
        done
      done
    done
  done;
  for i = 0 to Array.length grad.ws - 1 do
    for row = 0 to grad.ws.(i).rows - 1 do
      for col = 0 to grad.ws.(i).cols - 1 do
        Mat.set grad.ws.(i) row col
          ((-.(Mat.get grad.ws.(i) row col)) *. rate /. (Float.of_int tx.rows))
      done
    done;
    for row = 0 to grad.bs.(i).rows - 1 do
      for col = 0 to grad.bs.(i).cols - 1 do
        Mat.set grad.bs.(i) row col
          ((-.(Mat.get grad.bs.(i) row col)) *. rate /. (Float.of_int tx.rows))
      done
    done;
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
    Printf.printf "=== Test %i/%i ===\n" (t_row + 1) tx.rows;

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
