Please have a look at *Tutorial.ipynb*! However, if you want a quick preview of the outputs of the cells, you may visualize the HTML after execution of the cells: [tutorial_page.html](https://htmlpreview.github.io/?https://github.com/mistasse/mlworkflow/blob/master/tutorial_page.html) which is nececssary as dynamic widgets are not saved in *.ipynb* files.

## What is mlworkflow?

It is a library providing several separate modules to:

- Have an interactive feedback from your model being trained
- Aggregate data about your models and experiments, accessing them in a convenient manner
- Help you to obtain replayable and modifiable experiments
- Put notes and comments on them
- Make the use of your models more practical

That the modules are there do not mean you should use them. Only use them if they fit your way of experimenting. If you have other habits that you feel like would deserve a place in the library, we strongly encourage you to try to implement it in a generic fashion and to submit a pull-request, or to suggest it.

This library is developped at UCLouvain, and is mostly an attempt to make people feel that "tensorboard" does not answer to everything we may want about keeping tracks of our experiments. It is under MIT license.

## What is not currently handled that could frustrate users?

The interactive parts of the library (LivePanels, Dashboard) mostly rely on the use of Jupyter notebooks. However, a lot of people do not run their experiments in Jupyter notebooks. (Here, we use them as "front-ends" to our models, ...) So you may want to create summaries from Python scripts as well and only run the dashboard in a notebook. This problem will most likely be addressed. For the rest (Dataset, DataCollection, ...) nothing should rely on Jupyter. If it does, this should be fixed.

There is not much documentation and not all the features are commented yet, but we hope the tutorial may help you to get started with it, or why not getting you to ask for more.

Finally, remember this is best described as a side-project for accelerating our research that we thought may benefit other people.

## Installation

You can install it by typing

```bash
pip install mlworkflow
```

Or if expect to modify it, clone this repo and type

```
pip install -e .
```

in it.

Please note it may work on Python inferior to 3.6, but some of the tests do require 3.6 (dicts being ordered by default) we do not officialize it for now.