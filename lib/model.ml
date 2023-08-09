module Mat = Matrix

type t =
  { ws : Matrix.t array;
    bs : Matrix.t array;
    actf : Act_func.t;
  }

let random
      (shape: int array)
      ?(min=(-. 1.)) ?(max=1.)
      ?(actf=Act_func.sigmoid) (): t =
  let ws = Array.init (Array.length shape - 1)
             (fun i -> Mat.random shape.(i) shape.(i+1) ~min ~max ()) in
  let bs = Array.init (Array.length shape - 1)
             (fun i -> Mat.random 1 shape.(i+1) ~min ~max ()) in
  { ws = ws;
    bs = bs;
    actf = actf;
  }

let gradient_for_model (model: t): t =
  let ws = Array.init (Array.length model.ws)
             (fun i -> Mat.zeros model.ws.(i).rows model.ws.(i).cols) in
  let bs = Array.init (Array.length model.bs)
             (fun i -> Mat.zeros model.bs.(i).rows model.bs.(i).cols) in
  { ws = ws;
    bs = bs;
    actf = model.actf;
  }

let forward (model: t) (input: Mat.t): Mat.t =
  let ws = Array.to_seq model.ws in
  let bs = Array.to_seq model.bs in
  let fwd_layer x w b =
    Mat.map model.actf.f (Mat.sum (Mat.dot x w) b) in
  Seq.fold_left2 fwd_layer input ws bs

let cost (model: t) ~(tx: Mat.t) ~(ty: Mat.t): float =
  let sum = ref 0. in
  for t_row = 0 to tx.rows - 1 do
    let output = forward model (Mat.take_row tx t_row) in
    for y_col = 0 to ty.cols - 1 do
      let diff = (Mat.get ty t_row y_col) -.
                 (Mat.get output 0 y_col) in
      sum := !sum +. (Float.abs diff)
    done
  done;
  (!sum /. (Float.of_int (ty.cols * tx.rows)))

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
    Mat.map model.actf.f (Mat.sum (Mat.dot x w) b) in
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
        let act_out = Mat.get model_a.(a_i) 0 a_col in
        let first = 2. *. (Mat.get grad_a.(a_i) 0 a_col) *. (model.actf.d act_out) in
        Mat.set_with grad.bs.(a_i - 1) 0 a_col
          (fun x -> x +. first);
        for w_row = 0 to grad.ws.(a_i - 1).rows - 1 do
          Mat.set_with grad.ws.(a_i - 1) w_row a_col
            (fun x -> x +. (first *. (Mat.get model_a.(a_i - 1) 0 w_row)));
          Mat.set_with grad_a.(a_i - 1) 0 w_row
            (fun x -> x +. (first *. (Mat.get model.ws.(a_i - 1) w_row a_col)));
        done
      done
    done
  done;
  { model with
    ws = Array.map (Mat.map (fun x -> (-.x) *. rate /. (Float.of_int tx.rows))) grad.ws;
    bs = Array.map (Mat.map (fun x -> (-.x) *. rate /. (Float.of_int tx.rows))) grad.bs;
  }

let apply_gradient (model: t) (grad: t): t =
  { model with
    ws = Array.mapi (fun i x -> Mat.sum x grad.ws.(i)) model.ws;
    bs = Array.mapi (fun i x -> Mat.sum x grad.bs.(i)) model.bs;
  }

let evaluate (model: t) ~(tx: Mat.t) ~(ty: Mat.t): unit =
  let print_row (row: Mat.t): unit =
    Printf.printf "[ ";
    Array.iter (fun x -> Printf.printf "%f, " x) row.arr;
    Printf.printf "]\n";
  in
  for t_row = 0 to tx.rows - 1 do
    Printf.printf "=== Test %i/%i ===\n" (t_row + 1) tx.rows;

    Printf.printf "\tFor input: ";
    let input = Mat.take_row tx t_row in
    print_row input;

    Printf.printf "\tExpected: ";
    let expected = Mat.take_row ty t_row in
    print_row expected;

    Printf.printf "\tReceived: ";
    let output = forward model input in
    print_row output;

    let error_mat = Mat.map (fun x -> -. x) output
                    |> Mat.sum expected
                    |> Mat.map (fun x -> Float.abs x) in
    let error = Array.fold_left ( +. ) 0. error_mat.arr in
    Printf.printf "\tError: %f\n" error;
  done;
  Stdlib.flush Stdlib.stdout

let chunk_train_set (count: int) (nth: int) ((tx, ty): Mat.t * Mat.t): Mat.t * Mat.t =
  let max = ((count * (nth + 1)) + count) in
  let actual_count =
    if tx.rows <= max
    then count + ((tx.rows + count) - max)
    else count
  in

  let tx_arr = ref (Array.make 0 0.) in
  let ty_arr = ref (Array.make 0 0.) in
  for i = 0 to actual_count - 1 do
    tx_arr := Array.append !tx_arr (Mat.take_row tx ((count * nth) + i)).arr;
    ty_arr := Array.append !ty_arr (Mat.take_row ty ((count * nth) + i)).arr;
  done;
  { tx with
    arr = !tx_arr;
    rows = actual_count
  },
  { ty with
    arr = !ty_arr;
    rows = actual_count
  }
