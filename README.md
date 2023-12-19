# Recognizing Japanese Text

An open sourced Japanese digital writing classification model that makes predictions based on generated observations from drawing interaction from an android app.

<img src="mod4/project/img/hidden_layers/transferlearning_custom/customhl4_social.png" alt="Hidden layers"></img>

This project started out as an assignment with a deadline of only a few weeks that ended up being a race against time to squeeze in an MVP in the form of a CNN with an existing datastet while building my first Android app to facilitate creating a new dataset from scratch that would be used for my actual idea. If you are here to see the initial version and notebooks, those parts have been archived [here](/mod4project)

## The Original (probably but not necessarily abandoned) Roadmap:
**Stage 1** - Build an OCR recognition model using existing data from the Kuzushiji-49. The observations have a degree of separation from the goal of this project, but it also provides an advantages in terms of comparison/future generalizations in that it's classification is a more difficult task since the historical kuzushiji script is less standardized.<br>**Completed-6/16/2020**

**Stage 2** - Use transfer learning to bring the smaller dataset up to speed with the models of the Kuzushiji-49.<br>**Completed-6/16/2020**

**Stage 3** -Once there are significant observations build a standalone without the kuzushiji data and determine the best architecture for the OCR model.<br>**Completed-6/16/2020**

**Stage 4** -  Explore the notion of using the raw data to provide additional data captured (the bitmap images inherently do not capture stroke direction or order).<br>**Completed-6/16/2020**

**Stage 5** Rewrite a versatile study app that can allow generation of observations more efficiently.<br>**Not Started - TBA**

**Stage 6** Expand to the katakana and Kanji datasets.<br>**Not Started - TBA**
