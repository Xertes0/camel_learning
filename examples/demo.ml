open Camel_learning

type t =
  { shape : int array;
    tx : Matrix.t;
    ty : Matrix.t;
    rate : float;
    actf : Act_func.t;
  }

let run
      (demo: t)
      ?(goal=0.01)
      (): unit =
  Random.self_init ();
  let model = ref (Model.random demo.shape ~actf:demo.actf ()) in

  let last_cost = ref Float.max_float in
  let iter_count = ref 1 in
  while !last_cost > goal do
    (* let grad = Model.finite_difference !model ~tx:demo.tx ~ty:demo.ty ~eps:0.1 ~rate:demo.rate in *)
    let grad = Model.back_propagation !model ~tx:demo.tx ~ty:demo.ty ~rate:demo.rate in
    model := Model.apply_gradient !model grad;
    last_cost := Model.cost !model ~tx:demo.tx ~ty:demo.ty;

    (* Print cost only every 10_000 iterations *)
    if !iter_count mod 10_000 = 0 then (
      Printf.printf "Cost: %f\n" !last_cost;
      Stdlib.flush Stdlib.stdout;
    );
    iter_count := !iter_count + 1
  done;

  Printf.printf "Model reached the cost of %f after %i iterations.\n" !last_cost !iter_count;
  Model.evaluate !model ~tx:demo.tx ~ty:demo.ty
