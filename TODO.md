# Roadmap to 0.5.0 (or 1.0.0?) no, 0.5.0 is fine.

## Features
- [x] Better CLI support for custom models. The constructor currently cannot receive any parameter. This is very wrong.
- [ ] NNViz should be usable as a library. I need to turn this mess into a proper API.

## Docs
- [x] Full docstring coverage
- [ ] ReadTheDocs integration

## Tests
- [ ] Full test coverage
- [ ] More unit tests for the core functionality as now it's mostly integration tests

## Won't do
- [ ] Add more info to collapsed nodes. Currently, only the node name is shown. It would be nice to show the number of children, parameters, source, class, etc.
- [ ] Guessing the input to pass to a model is frustrating. It would be nice to have a silver-bullet input string that adapts to any given model. This would be a killer feature for the CLI.
