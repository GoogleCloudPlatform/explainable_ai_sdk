# Explainable AI SDK

This is an SDK for
[Google Cloud Explainable AI](https://cloud.google.com/explainable-ai) service.
Explainable AI SDK helps users build [explanation metadata](https://cloud.google.com/ai-platform/prediction/docs/ai-explanations/preparing-metadata) for their models and visualize feature
attributions returned from the model.

## Installation

Explainable AI SDK are available directly on [Google Cloud AI Platform Notebooks](https://cloud.google.com/ai-platform-notebooks).
For other platforms, you can install it via

```shell
pip install explainable-ai-sdk
```

Note that we require shell enviroment that has [Google Cloud SDK](https://cloud.google.com/sdk/docs/quickstarts) installed.
Otherwise, the SDK will not be able to communicate with the AI Platform.

## Usage

### Metadata Builders
Users of the library can create explanation metadata JSON files using metadata
builders. There are various metadata builders for Tensorflow V1 and V2 in their
respective folders.

#### Tensorflow v1
We provide three different builders catering to three main
Tensorflow interfaces: **Estimators**, **Keras models**, and **models built with
the low-level API (ops and tensors)**. An example usage for a Keras model would
be as follows:

```python
my_model = keras.models.Sequential()
my_model.add(keras.layers.Dense(32, activation='relu', input_dim=10))
my_model.add(keras.layers.Dense(32, activation='relu'))
my_model.add(keras.layers.Dense(1, activation='sigmoid'))
builder = explainable_ai_sdk.metadata.tf.v1.KerasGraphMetadataBuilder(my_model)
builder.get_metadata(). # To get a dictionary representation of the metadata.
builder.save_model_with_metadata('gs://my_bucket/model')  # Save the model and the metadata.
```

#### Tensorflow v2
There is a single metadata builder that takes a **saved model**.
Example usage would be as follows:

```python
builder = explainable_ai_sdk.metadata.tf.v2.SavedModelMetadataBuilder(
        model_path)
builder.get_metadata(). # To get a dictionary representation of the metadata.
builder.save_model_with_metadata('gs://my_bucket/model')  # Save the model and the metadata.
```

### Making Predict and Explain Calls
We offer a model interface to communicate with the deployed model more easily.
With this interface, users can call `predict()` and `explain()` functions to get
predictions and explanations for the provided data points, respectively.
Here is an example snippet for using the model interface:

```python
project_id = "example_project"
model_name = "example_model"
version_name = "v1"

m = explainable_ai_sdk.load_model_from_ai_platform(project_id, model_name, version_name)
instances = []

# ... steps for preparing instances

predictions = m.predict(instances)
explanations = m.explain(instances)
```

### Explanation, Attribution, and Visualization
What's returned from `explain()` function is a list of `Explanation` objects --
one `Explanation` per input instance. This object aims to make interactions with
returned attributions more easily. Here are a few usages of the `Explanation`
object.

**Note**: The `feature_importance` and `as_tensors` functions are only
working on tabular models now due to the limited payload size. We are working on
making it available on image models.**

#### Get feature importance

The `feature_importance` function will return the imporance of each feature
based on feature attributions. Note that if a feature has more than one
dimension (e.g., an image has RGB channels in each pixel, the importance is
calculated based on the aggregation).

```python
explanations[0].feature_importance()
```

#### Get raw attributions

If users want to get feature attributions over each dimension, `as_tensors`
function will return the raw attributions as tensors.

```python
explanations[0].as_tensors()
```

#### Visualize attributions

The `Explanation` class provides a way to let users visualize attributions
directly. Users can simply call `visualize_attributions` to see feature
attributions. This works for both image and tabular models.

```python
explantions[0].visualize_attributions()
```


## Caveats
* This library works with (and depends) on either major version of Tensorflow.
* `metadata/tf/v1` and `metadata/tf/v2` folders shouldn't be imported in the
same python runtime to avoid unintended side effects of mixing Tensorflow 1 and
2 behavior.

## License

All files in this repository are under the
[Apache License, Version 2.0](https://github.com/GoogleCloudPlatform/explainable_ai_sdk/blob/master/LICENSE)
unless noted otherwise.

**Note:** We are not accepting contributions at this time.
