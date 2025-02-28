# Comparative Study of Human Pose Estimation Algorithms for Clinical Applications

One Paragraph of the project description

## Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes. See deployment
for notes on deploying the project on a live system.

### Prerequisites

- For MediaPipe, download the MediaPipe Pose landmarker bundle you wish to use from
[here](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index#models)
and include it as ./configs/mediapipe/mediapipe_pose_landmarker.task

- For HRNet, download the pre-trained model from
[here](https://drive.google.com/file/d/1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38/view?usp=drive_link)
and include it as ./configs/hrnet/hrnet_pretrainied.pth.

### Installing

A step by step series of examples that tell you how to get a development
environment running

Say what the step will be

    Give the example

And repeat

    until finished

End with an example of getting some data out of the system or using it
for a little demo





Create development environment:

    conda create --name hpe_comparison python=3.9
    conda activate hpe_comparison
    cd "path to repository"

Install requirements with pip:

    pip install -r requirements.txt

If running on GPU:

    pip install torch torchvision

If running on CPU:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu







## Running the tests

Explain how to run the automated tests for this system

### Sample Tests

Explain what these tests test and why

    Give an example

### Style test

Checks if the best practices and the right coding style has been used.

    Give an example

## Deployment

Add additional notes to deploy this on a live system

## Built With

  - [Contributor Covenant](https://www.contributor-covenant.org/) - Used
    for the Code of Conduct
  - [Creative Commons](https://creativecommons.org/) - Used to choose
    the license

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions
available, see the [tags on this
repository](https://github.com/PurpleBooth/a-good-readme-template/tags).

## Authors

  - **Billie Thompson** - *Provided README Template* -
    [PurpleBooth](https://github.com/PurpleBooth)

See also the list of
[contributors](https://github.com/PurpleBooth/a-good-readme-template/contributors)
who participated in this project.

## License

This project is licensed under the [CC0 1.0 Universal](LICENSE.md)
Creative Commons License - see the [LICENSE.md](LICENSE.md) file for
details

## Acknowledgments

  - Hat tip to anyone whose code is used
  - Inspiration
  - etc
