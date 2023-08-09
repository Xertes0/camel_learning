open Camel_learning

;;
Demo_sdl.run
  { tx = Matrix.from_3d_list
           [ [ 0.; 0. ];
             [ 0.; 1. ];
             [ 1.; 0. ];
             [ 1.; 1. ] ];
    ty = Matrix.from_3d_list
           [ [ 0. ];
             [ 1. ];
             [ 1. ];
             [ 1. ] ];
    chunk_size = None;
    shape = [| 2; 1; |];
    rate = 1.;
    actf = Act_func.sigmoid;
  } ()
