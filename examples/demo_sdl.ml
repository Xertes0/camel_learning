open Tsdl
open Tsdl_ttf
open Camel_learning

type t =
  { shape : int array;
    tx : Matrix.t;
    ty : Matrix.t;
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

  let scale = 1. in
  let apply_scale = fun x -> x
                             |> Float.of_int
                             |> ( *. ) scale
                             |> Int.of_float in
  let x_spacing = apply_scale (screen_width / (Array.length model.ws)) in

  for w_i = 0 to Array.length model.ws - 1 do
    for row = 0 to model.ws.(w_i).rows - 1 do
      let row_y_spacing = apply_scale (screen_height / model.ws.(w_i).rows) in
      let col_y_spacing = apply_scale (screen_height / model.ws.(w_i).cols) in

      for col = 0 to model.ws.(w_i).cols - 1 do
        let color = Matrix.get model.ws.(w_i) row col
                    |> ( +. ) 1.
                    |> ( /. ) 2.
                    |> ( *. ) 255.
                    |> Int.of_float in
        Sdl.set_render_draw_color r
          0xFF color color 0xFF >>= fun () ->
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
  let iter_count = ref 0 in

  let quit = ref false in
  let evaluated = ref false in
  while !quit = false do
    iter_count := !iter_count + 1;

    (* Train model every time *)
    if !last_cost > goal then (
      let grad = Model.back_propagation !model ~tx:demo.tx ~ty:demo.ty ~rate:demo.rate in
      model := Model.apply_gradient !model grad;

      if !iter_count mod 10 = 0 then
        last_cost := Model.cost !model ~tx:demo.tx ~ty:demo.ty;
    ) else if !evaluated = false then (
      Printf.printf "Model reached the cost of %f after %i iterations.\n" !last_cost !iter_count;
      Model.evaluate !model ~tx:demo.tx ~ty:demo.ty;
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
