# 💾 Storing Graphs

## Supported formats

As you may have noticed, the `nnviz`, by default will save the drawn graph to a **pdf file** in the current working directory. This is the default behavior, but you can change it by passing the `-o` or `--out` option, followed by the path to the **output file**.

Full list of **supported formats** (see [Graphviz docs](https://graphviz.org/doc/info/output.html)) for more info:
`canon`, `cmap`, `cmapx`, `cmapx_np`, `dia`, `dot`, `fig`, `gd`, `gd2`, `gif`, `hpgl`, `imap`, `imap_np`, `ismap`, `jpe`, `jpeg`, `jpg`, `mif`, `mp`, `pcl`, `pdf`, `pic`, `plain`, `plain-ext`, `png`, `ps`, `ps2`, `svg`, `svgz`, `vml`, `vmlz`, `vrml`, `vtx`, `wbmp`, `xdot`, `xlib`

Examples:
```bash
nnviz resnet18 -o my_resnet18.pdf
nnviz resnet18 -o my_resnet18.png
nnviz resnet18 -o my_resnet18.svg
nnviz resnet18 --out /very/custom/path/to/wow.pdf
```

## Save/Load to JSON

If you plan to create multiple visualizations of the same model, you can use **JSON files** to store the graph representation of the model, and then load it from the file instead of performing a symbolic trace every time, which, if done repeatedly, can become quite slow.

To dump the graph representation to a JSON file, you can pass the `-j` or `--json` flag to the `nnviz` command.
The JSON file will be saved together with the pdf file, just with a ".json" extension instead of ".pdf".
```bash
nnviz resnet18 -j
nnviz resnet18 --json
nnviz resnet18 -j -o /very/custom/path/to/wow.pdf
```

In the third example (above), the JSON file will be saved to `/very/custom/path/to/wow.json`.

Ok, but once you have the JSON file, how do you load it? Well, you can pass the path to the JSON file to the `nnviz` command as if it was a model name, and it will **load the graph representation** from the file instead of performing a trace, and it will be blazing fast.
```bash
nnviz /very/custom/path/to/wow.json
```