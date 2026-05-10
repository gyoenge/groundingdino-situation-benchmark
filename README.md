# Net Challenge Project - **Dataset Process** 

## GroundingDINO Finetuning Dataset Processing/Preparing  

This repository contains **benchmark construction code for evaluating prompt-based situation understanding capabilities**, data preprocessing scripts, and knife safety detection dataset generation codes.

<!-- This repository contains the **code used to build the custom dataset for fine-tuning the GroundingDINO model**.  -->

<!-- About
A project from the K-Digital Net Challenge (2023). This repository contains benchmark construction code for evaluating prompt-based situation understanding capabilities, data preprocessing scripts, and knife safety detection dataset generation codes. -->

<!-- The main code for this project is located in the [GroundingDINO Finetune](https://github.com/gyoenge/net-challenge-groundingdino-finetune) repository, based on [Original GroundingDINO Finetune Pipeline Opensource](https://github.com/Asad-Ismail/Grounding-Dino-FineTuning). Please visit it for more details about the project.  -->

## Generation of Datasets

We aim to generate datsets for knife object detection task. It could be contain caption with slight detailed with person's motion (e.g. a person is holding a knife, a person is swinging a knife, ...). 

See the detailed information in `./gen_datasets/` and inside `README.md` file. 

- Example data: 

    <div align="center">
    <table>
        <tr>
        <td><img src=".assets/image_000005.jpg" width="250"/></td>
        <td><img src=".assets/image_000014.jpg" width="250"/></td>
        <td><img src=".assets/image_000017.jpg" width="250"/></td>
        </tr>
        <tr>
        <td><img src=".assets/image_000031.jpg" width="250"/></td>
        <td><img src=".assets/image_000118.jpg" width="250"/></td>
        <td><img src=".assets/image_000144.jpg" width="250"/></td>
        </tr>
    </table>
    </div>
