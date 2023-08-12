open Camel_learning

module Mat = Matrix

let main () =
  print_endline "Read README for information on this model.";
  if Array.length Sys.argv < 2 then (
    Printf.printf "Supply the path to the directory containing uncompressed mnist databases.\n";
    Printf.printf "You can get them from 'http://yann.lecun.com/exdb/mnist/'.\n";
    Stdlib.exit 1
  );
  let dir_path = Sys.argv.(1) in

  let mnist = Mnist_common.mnist_from_path dir_path in
  Printf.printf "Read %ix%i training images.\n" mnist.tx.rows mnist.tx.cols;
  Printf.printf "Read %ix%i training labels.\n" mnist.ty.rows mnist.ty.cols;
  Printf.printf "Read %ix%i testing images.\n" mnist.ver_x.rows mnist.ver_x.cols;
  Printf.printf "Read %ix%i testing labels.\n" mnist.ver_y.rows mnist.ver_y.cols;

  let interrupted = ref false in
  Sys.set_signal Sys.sigint (Sys.Signal_handle (fun _ -> interrupted := true));

  Random.self_init ();
  let shape = [| 784; 128; 10; |] in
  let model = ref (Model.random shape ~actf:Act_func.sigmoid ()) in

  let print_cost () =
    let (rand_tx, rand_ty) = Model.chunk_train_set 2 0 (mnist.tx, mnist.ty) in
    Printf.printf "Cost of train data: %f\n" (Model.cost !model ~tx:rand_tx ~ty:rand_ty);

    let (rand_tx, rand_ty) = Model.chunk_train_set 2 0 (mnist.ver_x, mnist.ver_y) in
    Printf.printf "Cost of unseen data: %f\n" (Model.cost !model ~tx:rand_tx ~ty:rand_ty);

    Stdlib.flush Stdlib.stdout;
  in

  print_cost ();
  let iter = ref 0 in
  while !interrupted = false do
    iter := !iter + 1;

    let (rand_tx, rand_ty) = Model.chunk_train_set 64 (!iter mod (mnist.tx.rows / 64)) (mnist.tx, mnist.ty) in
    let grad = Model.back_propagation !model ~tx:rand_tx ~ty:rand_ty ~rate:1. in
    model := Model.apply_gradient !model grad;

    if !iter mod 10 = 0 then
      print_cost ();
  done;
  print_endline "Interrupted\n";

  let (rand_tx, rand_ty) = Model.chunk_train_set 10 0 (mnist.ver_x, mnist.ver_y) in
  Model.evaluate !model ~tx:rand_tx ~ty:rand_ty;
  Model.save_to_file !model "mnist.model";

;;
main ()
