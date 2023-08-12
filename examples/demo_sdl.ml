open Camel_learning
open Tsdl
open Tsdl_ttf

type t =
  { shape : int array;
    tx : Matrix.t;
    ty : Matrix.t;
    chunk_size: int option;
    rate : float;
    actf : Act_func.t;
  }

let screen_width = 1280
let screen_height = 720

let last_cost = ref Float.max_float
let font_color = Sdl.Color.create ~r:0xEF ~g:0xEF ~b:0xEF ~a:255

let ( >>= ) x f =
  match x with
  | Error (`Msg e) -> failwith (Printf.sprintf "Error %s" e)
  | Ok a -> f a

let draw
      (r: Sdl.renderer)
      (model: Model.t)
      ~(font: Ttf.font): unit =
  Sdl.set_render_draw_color r
    0x0F 0x0F 0x0F 0xFF >>= fun () ->
  Sdl.render_clear r >>= fun () ->

  let value_to_color =
    fun x ->
    x
    |> ( *. ) (255. /. 3.)
    |> Int.of_float
    |> Int.abs in

  let scale = 0.8 in
  let apply_scale = fun x -> x
                             |> Float.of_int
                             |> ( *. ) scale
                             |> Int.of_float in
  let x_spacing = apply_scale (screen_width / (Array.length model.ws)) in

  for w_i = 0 to Array.length model.ws - 1 do
    let row_y_spacing = apply_scale (screen_height / model.ws.(w_i).rows) in
    let col_y_spacing = apply_scale (screen_height / model.ws.(w_i).cols) in
    let rect_size = col_y_spacing / 2 in

    for col = 0 to model.ws.(w_i).cols - 1 do
      let bcolor = value_to_color (Matrix.get model.bs.(w_i) 0 col) in
      Sdl.set_render_draw_color r
        (Int.max 0 (bcolor - 0xFF)) (Int.max 0 (bcolor - (0xFF * 2))) (Int.min 0xFF bcolor) (Int.min 0xFF bcolor) >>= fun () ->
      let brect = Sdl.Rect.create
                    ~x:(((w_i + 1) * x_spacing) - (rect_size / 2))
                    ~y:((col * col_y_spacing) + (col_y_spacing / 2) - (rect_size / 2))
                    ~w:rect_size ~h:rect_size in
      Sdl.render_fill_rect r (Some brect) >>= fun () ->

      for row = 0 to model.ws.(w_i).rows - 1 do
        let wcolor = value_to_color (Matrix.get model.ws.(w_i) row col) in
        Sdl.set_render_draw_color r
          (Int.max 0 (wcolor - 0xFF)) (Int.max 0 (wcolor - (0xFF * 2))) (Int.min 0xFF wcolor) (Int.min 0xFF wcolor) >>= fun () ->
        Sdl.render_draw_line r
          (w_i * x_spacing) ((row * row_y_spacing) + (row_y_spacing / 2))
          ((w_i + 1) * x_spacing) ((col * col_y_spacing) + (col_y_spacing / 2)) >>= fun () ->
        ()
      done;
    done;
  done;

  let text_str = Printf.sprintf "Cost: %f" !last_cost in
  Ttf.render_text_solid font text_str font_color >>= fun text_surf ->
  Sdl.create_texture_from_surface r text_surf >>= fun text_text ->

  Ttf.size_text font text_str >>= fun (text_x, text_y) ->
  let text_rect = Sdl.Rect.create ~x:(screen_width - text_x) ~y:(screen_height - text_y) ~w:text_x ~h:text_y in
  Sdl.render_copy r text_text ~dst:text_rect >>= fun () ->

  Sdl.render_present r;
  Sdl.delay 1l

let window_loop
      ~(renderer: Sdl.renderer)
      ~(demo: t)
      ~(goal: float) =
  let e = Sdl.Event.create () in

  Ttf.open_font "./font.ttf" 32 >>= fun font ->

  let model = ref (Model.random demo.shape ~actf:demo.actf ()) in
  (* model := Model.load_from_file !model "save.model"; *)
  let iter_count = ref 0 in

  let quit = ref false in
  let evaluated = ref false in
  while !quit = false do
    iter_count := !iter_count + 1;

    (* Train model every time *)
    if !last_cost > goal then (
      (match demo.chunk_size with
       | Some size ->
          for i = 0 to (demo.tx.rows / size) - 1 do
            let (tx, ty) = Model.chunk_train_set size i (demo.tx, demo.ty) in
            let grad = Model.back_propagation !model ~tx ~ty ~rate:demo.rate in
            model := Model.apply_gradient !model grad;
          done;
       | None ->
          let grad = Model.back_propagation !model ~tx:demo.tx ~ty:demo.ty ~rate:demo.rate in
          model := Model.apply_gradient !model grad; );

      if !iter_count mod 10 = 0 then
        last_cost := Model.cost !model ~tx:demo.tx ~ty:demo.ty;
    ) else if !evaluated = false then (
      Printf.printf "Model reached the cost of %f after %i iterations.\n" !last_cost !iter_count;
      Model.evaluate !model ~tx:demo.tx ~ty:demo.ty;
      (* Model.save_to_file !model "save.model"; *)
      evaluated := true;
    );

    (* Redraw window every 1_000 iterations *)
    if !iter_count mod 10 = 0 || !last_cost < goal then (
      while Sdl.poll_event (Some e) do
        match Sdl.Event.(enum (get e typ)) with
        | `Quit -> quit := true
        | _ -> ()
      done;
      draw renderer !model ~font
    );
  done

let run
      (demo: t)
      ?(goal=0.01)
      (): unit =
  Random.self_init ();

  Sdl.init Sdl.Init.(video + events) >>= fun () ->
  Ttf.init () >>= fun () ->

  Sdl.create_window ~w:screen_width ~h:screen_height "Camel Learning demo" Sdl.Window.shown >>= fun window ->
  Sdl.create_renderer window ~flags:Sdl.Renderer.accelerated >>= fun renderer ->

  window_loop ~renderer ~demo ~goal;
  Sdl.destroy_renderer renderer;
  Sdl.destroy_window window;
  Ttf.quit ();
  Sdl.quit ()
