open Camel_learning
open Tsdl
open Tsdl_ttf

let screen_width = 1280
let screen_height = 720

let font_color = Sdl.Color.create ~r:0xEF ~g:0xEF ~b:0xEF ~a:255
let mnist_text_scale = 10
let mnist_text_size = 28 * mnist_text_scale
let mnist_text_offset = 20

let ( >>= ) x f =
  match x with
  | Error (`Msg e) -> failwith (Printf.sprintf "Error %s" e)
  | Ok a -> f a

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

let update_mnist_texture (text: Sdl.texture) (row: Matrix.t): unit =
  let mnist_ba =
    Bigarray.Array1.init
      Bigarray.int32
      Bigarray.c_layout
      (28 * 28)
      (fun i ->
        let pixel = Matrix.get row 0 i
                    |> ( *. ) 255.
                    |> Int32.of_float in
        Int32.logor pixel (Int32.logor (Int32.shift_left pixel 8) (Int32.shift_left pixel 16))) in
  Sdl.update_texture text None mnist_ba 28 >>= fun () -> ()

type sidebar_el =
  { dst_rect : Sdl.rect;
    text: Sdl.texture;
  }

let make_sidebar ~(renderer: Sdl.renderer) ~(font: Ttf.font) ~(curr_out: Matrix.t array): sidebar_el list =
  Seq.zip (List.to_seq [0;1;2;3;4;5;6;7;8;9]) (Array.to_seq curr_out.(Array.length curr_out - 1).arr)
  |> List.of_seq
  |> List.sort (fun (_, x1) (_, x2) -> compare x2 x1)
  |> List.mapi (fun j (i, x) ->
         let text_str = Printf.sprintf "%i: %f" i (x *. 100.) in
         Ttf.render_text_solid font text_str font_color >>= fun text_surf ->
         Sdl.create_texture_from_surface renderer text_surf >>= fun text_text ->
         Sdl.free_surface text_surf;

         Ttf.size_text font text_str >>= fun (text_x, text_y) ->
         let text_rect = Sdl.Rect.create
                           ~x:(screen_width - text_x - mnist_text_offset) ~y:((28*10) + (mnist_text_offset * 2) + (text_y * j))
                           ~w:text_x ~h:text_y in
         { dst_rect = text_rect;
           text = text_text;
         }
       )

type vis_state =
  { renderer : Sdl.renderer;
    font : Ttf.font;
    model : Model.t;
    curr_in : Matrix.t;
    (* curr_out : Matrix.t array; *)
    mnist_text : Sdl.texture;
    model_surf : Sdl.surface;
    model_r : Sdl.renderer;
    model_text : Sdl.texture;
    sidebar : sidebar_el list;
  }

let new_model_texture (state: vis_state): Sdl.texture =
  Sdl.destroy_texture state.model_text;
  Sdl.create_texture_from_surface state.renderer state.model_surf >>= fun text -> text

let new_vis_input (state: vis_state) (inp: Matrix.t): vis_state =
  let new_out = Model.forward_collect state.model inp in

  update_mnist_texture state.mnist_text inp;

  render_model state.model new_out state.model_r;
  let new_model_text = new_model_texture state in

  let new_sidebar = make_sidebar ~renderer:state.renderer ~font:state.font ~curr_out:new_out in
  List.iter (fun el ->
      Sdl.destroy_texture el.text
    ) state.sidebar;

  { state with
    curr_in = inp;
    model_text = new_model_text;
    sidebar = new_sidebar;
  }

let make_state ~(renderer: Sdl.renderer) ~(font: Ttf.font) ~(model: Model.t) ~(inp: Matrix.t): vis_state =
  (* Mnist image *)
  Sdl.create_texture
    renderer
    Sdl.Pixel.format_rgb888 Sdl.Texture.access_streaming
    ~w:28 ~h:28 >>= fun mnist_text ->
  update_mnist_texture mnist_text inp;

  (* Model *)
  let curr_out = Model.forward_collect model inp in

  Sdl.create_rgb_surface
    ~w:screen_width ~h:screen_height ~depth:32
    0l 0l 0l 0l >>= fun model_surf ->

  Sdl.create_software_renderer model_surf >>= fun model_r ->
  render_model model curr_out model_r;
  let model_text = Sdl.create_texture_from_surface renderer model_surf >>= fun text -> text in

  (* Sidebar *)
  let sidebar = make_sidebar ~renderer ~font:font ~curr_out in

  { renderer = renderer;
    font = font;
    model = model;
    curr_in = inp;
    mnist_text = mnist_text;
    model_surf = model_surf;
    model_r = model_r;
    model_text = model_text;
    sidebar = sidebar;
  }

let window_loop (r: Sdl.renderer) =
  let e = Sdl.Event.create () in
  Ttf.open_font "./font.ttf" 24 >>= fun font ->

  let mnist = Mnist_common.mnist_from_path Sys.argv.(1) in
  let state = ref (make_state
                     ~renderer:r
                     ~font
                     ~model:(Model.load_from_file (Model.random [| 784; 128; 10; |] ~actf:Act_func.sigmoid ()) "mnist.model")
                     ~inp:(Matrix.take_row mnist.ver_x 0)) in

  let mouse_but_down = ref `None in
  let mouse_motion (mousex: int) (mousey: int) =
    let x = (mousex - (screen_width - mnist_text_size - mnist_text_offset)) / mnist_text_scale in
    let y = (mousey - mnist_text_offset) / mnist_text_scale in
    if x >= 0 && y >= 0 && x < 28 && y < 28
    then (
      let color = if !mouse_but_down = `Left then 1.0 else 0.0 in
      let put x y mul =
        if x >= 0 && y >= 0 && x < 28 && y < 28
        then Matrix.set_with !state.curr_in 0 ((y * 28) + x)
               (fun x -> if x < (color *. mul) || color = 0. then (color *. mul) else x)
      in

      put x y 1.0;
      put (x+1) y 0.3;
      put (x-1) y 0.6;
      put x (y+1) 0.5;
      put x (y-1) 0.7;

      update_mnist_texture !state.mnist_text !state.curr_in
    )
  in

  let last_set_row = ref 0 in
  let set_image_from_trainset (row: int) =
    state := new_vis_input !state (Matrix.take_row mnist.ver_x row);
  in

  let quit = ref false in
  while !quit = false do
    while Sdl.poll_event (Some e) do
      match Sdl.Event.(enum (get e typ)) with
      | `Quit -> quit := true
      | `Key_down -> (
        match Sdl.(get_key_name Event.(get e keyboard_keycode)) with
        | "Left" -> last_set_row := !last_set_row - 1;
                    if !last_set_row < 0
                    then last_set_row := mnist.ver_x.rows - 11;
                    set_image_from_trainset !last_set_row
        | "Right" -> last_set_row := !last_set_row + 1;
                     if !last_set_row = mnist.ver_x.rows
                     then last_set_row := 0;
                     set_image_from_trainset !last_set_row
        | "Up" -> set_image_from_trainset (Random.int mnist.ver_x.rows)
        | "Down" -> state := new_vis_input !state (Matrix.zeros 1 (28*28))
        | _ -> ()
      )
      | `Mouse_button_down -> (
        mouse_but_down :=
          match Sdl.Event.(get e mouse_button_button) with
          | 1 -> `Left
          | 3 -> `Right
          | _ -> !mouse_but_down
      )
      | `Mouse_button_up ->
         if !mouse_but_down != `None
         then (
           state := new_vis_input !state !state.curr_in
         );
         mouse_but_down := `None
      | `Mouse_motion ->
         if !mouse_but_down != `None
         then mouse_motion Sdl.Event.(get e mouse_motion_x) Sdl.Event.(get e mouse_motion_y)
      | _ -> ()
    done;

    Sdl.set_render_draw_color r
      0x0F 0x0F 0x0F 0xFF >>= fun () ->
    Sdl.render_clear r >>= fun () ->

    (* Model *)
    Sdl.render_copy r !state.model_text >>= fun () ->

    (* Mnist image *)
    let mnist_text_dst = Sdl.Rect.create
                           ~x:(screen_width - mnist_text_size - mnist_text_offset) ~y:mnist_text_offset
                           ~w:mnist_text_size ~h:mnist_text_size in
    Sdl.render_copy r !state.mnist_text ~dst:mnist_text_dst >>= fun () ->

    (* Sidebar *)
    List.iter (fun el ->
        Sdl.render_copy r el.text ~dst:el.dst_rect >>= fun () -> ()
      ) !state.sidebar;

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
