# Interview Questions

Questions an admissions committee might ask about this project.
No answers are provided; these are prompts for preparation.

---

1. Your lesion-grouped split prevents the same lesion from appearing in
   both train and test. What is `lesion_id` in HAM10000, and could there
   still be leakage at a finer level?

2. You show a leakage comparison between image-mode and lesion-mode splits.
   What does a higher macro-F1 in image mode actually mean, and how do you
   interpret the size of the gap?

3. Why is macro-F1 the primary metric here rather than accuracy or
   weighted-F1?

4. HAM10000 is heavily imbalanced. Why did you choose class-weighted
   cross-entropy rather than oversampling or a focal loss?

5. Your class weights use the formula N / (C * n_c). Where does this
   formula come from, and what assumptions does it make?

6. The checkpoint fingerprint includes the manifest SHA-256. What problem
   does that solve, and what would go wrong without it?

7. Your bootstrap resamples lesion clusters rather than individual images.
   What assumption does that address, and what does it leave open?

8. Grad-CAM highlights high-gradient regions at one convolutional layer.
   What can you conclude from a heatmap that falls entirely on the lesion?
   What can you conclude from one that falls off the lesion?

9. You chose ResNet-18 layer4[-1] and EfficientNet-B0 features[-1] as
   Grad-CAM target layers. Why those layers, and what is the trade-off
   between choosing an earlier versus a later layer?

10. Your training time is measured around the epoch loop only. What
    pre-processing or setup time does that exclude, and does that matter
    for comparing models?

11. The rarest classes (df, vasc) have very few lesions in the test set.
    What does that mean for the reliability of their recall numbers?

12. HAM10000 was collected at two dermatology centres. How would you
    expect these models to generalise to images from a different clinic
    or imaging device?

13. You use early stopping on validation macro-F1. What are the risks of
    this selection criterion, and what would you do differently?

14. The project has no hyperparameter search. What is the reason for
    that, and what would be the minimum rigorous way to add one?

15. What is the single most important limitation of this evaluation, and
    how would you address it with more time or data?
