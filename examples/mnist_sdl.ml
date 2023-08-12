(* TODO: Move repeated definitions to a separate file *)

open Camel_learning
open Tsdl
open Tsdl_ttf

let screen_width = 1280
let screen_height = 720

let font_color = Sdl.Color.create ~r:0xEF ~g:0xEF ~b:0xEF ~a:255

let ( >>= ) x f =
  match x with
  | Error (`Msg e) -> failwith (Printf.sprintf "Error %s" e)
  | Ok a -> f a

let set_mnist_image (ba: ('a, 'b, 'c) Bigarray.Array1.t) (row: Matrix.t): unit =
  for i = 0 to (28 * 28) - 1 do
    let pixel = Matrix.get row 0 i
                |> ( *. ) 255.
                |> Int32.of_float in
    ba.{i} <- Int32.logor pixel (Int32.logor (Int32.shift_left pixel 8) (Int32.shift_left pixel 16));
  done

let render_model (model: Model.t) (act: Matrix.t array) (r: Sdl.renderer): unit =
  Sdl.set_render_draw_color r
    0x0F 0x0F 0x0F 0xFF >>= fun () ->
  Sdl.render_clear r >>= fun () ->

  let screen_widthf = Float.of_int screen_width in
  let screen_heightf = Float.of_int screen_height in

  let value_to_color =
    fun x ->
    x
    |> ( *. ) (255. /. 3.)
    |> Int.of_float
    |> Int.abs in

  let scalex = ( *. ) 0.75 in
  let scaley = ( *. ) 1.0 in
  let x_spacing = scalex (screen_widthf /. Float.of_int (Array.length model.ws)) in

  for w_i = 0 to Array.length model.ws - 1 do
    let col_y_spacing = scaley (screen_heightf /. Float.of_int model.ws.(w_i).cols) in
    let rectw = x_spacing /. Float.of_int model.ws.(w_i).rows in
    let recth = col_y_spacing in

    for col = 0 to model.ws.(w_i).cols - 1 do
      for row = 0 to model.ws.(w_i).rows - 1 do
        let color = value_to_color
                      ((Matrix.get model.ws.(w_i) row col) *. (Matrix.get act.(w_i + 1) 0 col *. 7.5)) in
        Sdl.set_render_draw_color r
          (Int.max 0 (color - 0xFF)) (Int.max 0 (color - (0xFF * 2))) (Int.min 0xFF color) (Int.min 0xFF color) >>= fun () ->
        let rect = Sdl.Rect.create
                      ~x:(Int.of_float (((Float.of_int w_i) *. x_spacing) +. ((Float.of_int row) *. rectw)))
                      ~y:(Int.of_float (Float.of_int col *. col_y_spacing))
                      ~w:(Int.of_float rectw) ~h:(Int.of_float recth) in
        Sdl.render_fill_rect r (Some rect) >>= fun () -> ()
      done
    done
  done;
  Sdl.render_present r

let window_loop (r: Sdl.renderer) =
  let e = Sdl.Event.create () in
  Ttf.open_font "./font.ttf" 24 >>= fun font ->

  Sdl.create_texture
    r
    Sdl.Pixel.format_rgb888 Sdl.Texture.access_streaming
    ~w:28 ~h:28 >>= fun mnist_text ->
  let mnist_ba = Bigarray.Array1.create
                   Bigarray.int32
                   Bigarray.c_layout
                   (28 * 28) in

  let mnist = Mnist_common.mnist_from_path Sys.argv.(1) in
  let shape = [| 784; 128; 10; |] in
  let model = ref (Model.random shape ~actf:Act_func.sigmoid ()) in
  model := Model.load_from_file !model "mnist.model";

  let image_row = ref 0 in
  let output = ref (Model.forward_collect !model (Matrix.take_row mnist.ver_x !image_row)) in
  set_mnist_image mnist_ba (Matrix.take_row mnist.ver_x !image_row);

  Sdl.create_rgb_surface
    ~w:screen_width ~h:screen_height ~depth:32
    0l 0l 0l 0l >>= fun model_surf ->
  Sdl.create_software_renderer model_surf >>= fun model_r ->
  render_model !model !output model_r;
  let model_text = ref (Sdl.create_texture_from_surface r model_surf >>= fun text -> text) in

  let update_image () =
    let row = (Matrix.take_row mnist.ver_x !image_row) in
    output := Model.forward_collect !model row;
    set_mnist_image mnist_ba row;

    render_model !model !output model_r;
    model_text := Sdl.create_texture_from_surface r model_surf >>= fun text -> text;
  in

  let quit = ref false in
  while !quit = false do
    while Sdl.poll_event (Some e) do
      match Sdl.Event.(enum (get e typ)) with
      | `Quit -> quit := true
      | `Key_down -> (
         match Sdl.(get_key_name Event.(get e keyboard_keycode)) with
         | "Left" -> image_row := !image_row - 1;
                     if !image_row < 0
                     then image_row := mnist.ver_x.rows - 11;
                     update_image ()
         | "Right" -> image_row := !image_row + 1;
                      if !image_row = mnist.ver_x.rows
                      then image_row := 0;
                      update_image ()
         | "Up" -> image_row := Random.int mnist.ver_x.rows;
                   update_image ()
         | _ -> ()
      )
      | _ -> ()
    done;

    Sdl.set_render_draw_color r
      0x0F 0x0F 0x0F 0xFF >>= fun () ->
    Sdl.render_clear r >>= fun () ->

    Sdl.render_copy r !model_text >>= fun () ->

    Sdl.update_texture mnist_text None mnist_ba 28 >>= fun () ->
    let mnist_text_size = 28*10 in
    let mnist_text_dst = Sdl.Rect.create
                           ~x:(screen_width - mnist_text_size - 20) ~y:20
                           ~w:mnist_text_size ~h:mnist_text_size in
    Sdl.render_copy r mnist_text ~dst:mnist_text_dst >>= fun () ->

    let eval_str_pos = ref 0 in
    Seq.zip (List.to_seq [0;1;2;3;4;5;6;7;8;9]) (Array.to_seq !output.(Array.length !output - 1).arr)
    |> List.of_seq
    |> List.sort (fun (_, x1) (_, x2) -> compare x2 x1)
    |> List.iter (fun (i, x) ->
           let text_str = Printf.sprintf "%i: %f" i (x *. 100.) in
           Ttf.render_text_solid font text_str font_color >>= fun text_surf ->
           Sdl.create_texture_from_surface r text_surf >>= fun text_text ->
           Ttf.size_text font text_str >>= fun (text_x, text_y) ->
           let text_rect = Sdl.Rect.create
                             ~x:(screen_width - text_x - 20) ~y:((28*10) + 40 + (text_y * !eval_str_pos))
                             ~w:text_x ~h:text_y in
           Sdl.render_copy r text_text ~dst:text_rect >>= fun () ->
           eval_str_pos := !eval_str_pos + 1
         );

    Sdl.render_present r;
    Sdl.delay 1l
  done

let main () =
  if Array.length Sys.argv < 2 then (
    Printf.printf "Supply the path to the directory containing uncompressed mnist databases.\n";
    Printf.printf "You can get them from 'http://yann.lecun.com/exdb/mnist/'.\n";
    Stdlib.exit 1
  );
  Random.self_init ();

  Sdl.init Sdl.Init.(video + events) >>= fun () ->
  Ttf.init () >>= fun () ->

  Sdl.create_window ~w:screen_width ~h:screen_height "Camel Learning demo" Sdl.Window.shown >>= fun window ->
  Sdl.create_renderer window ~flags:Sdl.Renderer.accelerated >>= fun renderer ->

  window_loop renderer;
  Sdl.destroy_renderer renderer;
  Sdl.destroy_window window;
  Ttf.quit ();
  Sdl.quit ()

;;
main ()
