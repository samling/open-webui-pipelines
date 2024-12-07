## Open Web-UI Functions, Tools, and Pipelines

### Overview

This repo is a collection of functions, tools, and pipelines for Open-WebUI. The differences between these three are as follows:

* **Functions** are built-in, modular operations that enhance the capabilities of an AI model by extending their functionality. They are installed into and executed from the Open-WebUI container directly. There are different types of functions:
    * **Filters** modify the user input/output to add, remove, or modify the content of the body object
        * **Inlets** pre-process the user input before sending to the model
        * **Outlets** post-process the response from the model
    * **Actions** add functionality to the message UI
    * **Pipes** create a model(s) with custom logic and processing
        * **Pipe** creates a single model
        * **Manifold** pipes are used to create a collection of Pipes

* **Tools** are python scripts that can be provided to an LLM at the time of the request. Tools let LLMs perform additional actions and receive additional context as the result. Like functions, they are installed into and executed from the Open-WebUI container directly.

* **Pipelines** are similar to functions, except they are installed into and executed from a dedicated pipelines container. The main benefit is the ability to offload the processing of the pipeline logic to a separate container, which may have more resources available to it, be connected to other systems, or contain additional dependencies.
    * **Pipelines** contain the same subtypes as **Functions**: **filters** and **pipes**. The method signatures differ from functions to pipelines however, so porting a function to a pipeline is not simply a matter of where it's installed. At the time of this writing (December 2024), certain functionality like emitters also do not exist in pipelines.

### What's in this repo?

This repo contains my own experiments of the above types of Open-WebUI additions. Some are functional, some are not. I've tried to organize and/or name them accordingly, but use at your own risk.

### Notes

* For development purposes I've included copies of the `utils` libraries present in both the open-webui and pipelines containers. While they're functionally similar, they're located in (and named) different things in their respective containers, hence they are included twice.