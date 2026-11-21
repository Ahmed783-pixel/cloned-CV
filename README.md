# Kinematics Labelling of Surgical Video Datasets

## Goal

Develop a vision-based system that annotates previously unlabelled endoscopic videos with **surgical instrument tip locations and estimated physical velocities**. The intended pipeline learns from video sequences paired with tip-location and velocity labels, then predicts these quantities on new videos without requiring robot kinematics at inference time.

Our current work establishes tip localization and tracking. The next stage is to prepare physically scaled velocity supervision and train a temporal velocity predictor. The target units are **millimetres per second (mm/s)** in the camera frame; current tracking-derived image velocities are **pixels per second**.

## Current progress

| Component | Status |
| --- | --- |
| YOLO Pose tip localization | Implemented: fine-tuning for surgical instrument landmarks. |
| Instrument tracking | Implemented experiments associating detections across frames and calculating image-plane velocity. |
| SurgPose adaptation | Tip-localization fine-tuning and stereo detection overlays documented in the upstream `Will` branch. |
| Physical velocity labels | Planned preparation from stereo annotations, calibration and verified acquisition timing; not verified in the inspected experiment code. |
| Learned velocity prediction | Not trained yet. |
| Physical velocity evaluation on unlabelled endoscopic video | Future application; accuracy requires separate labelled evaluation data. |

**YOLO has been trained for detection and keypoint localization, not velocity regression.** The existing pixel-velocity calculation is post-processing of tracked detections.

The project experiments inspected during development are in the [upstream course repository](https://github.com/yxLuo06/Computer-Vison-Course-project):

- [EndoVis experiment](https://github.com/yxLuo06/Computer-Vison-Course-project/tree/main/experiments/endovis-pose): YOLO26n-Pose predicts instrument boxes and five landmarks. The tracker uses the midpoint between the clasper landmarks as its tip.
- [SurgPose experiment on the Will branch](https://github.com/yxLuo06/Computer-Vison-Course-project/tree/Will/experiments/surgpose-tip): adapts the detector to jaw-tip annotations in left and right camera views. The stereo overlay visualizes detections; it does not itself reconstruct 3D positions or calculate mm/s labels.

## Proposed pipeline

### 1. Localize and associate instrument tips

Use the trained YOLO Pose model to detect instruments and localize tip keypoints. Associate detections across frames to retain a persistent instrument ID.

Choose a consistent tip definition before generating velocity labels: either an individual jaw endpoint or the midpoint of both jaw endpoints. The EndoVis tracker uses a midpoint, whereas the SurgPose experiment supervises individual jaw tips. A bounding-box centre is not a substitute for the tip.

### 2. Prepare physical velocity supervision

Use corresponding **dataset-annotated keypoints** in synchronized SurgPose left/right images and stereo calibration to reconstruct camera-frame positions. Handle lens distortion and verify calibration translation units before expressing positions in millimetres. For a jaw midpoint, reconstruct both endpoints and then calculate their 3D midpoint.

Verify actual acquisition timestamps or capture intervals before calculating velocity. An exported video's playback FPS, including 30 FPS, does not by itself establish the physical interval between captured samples.

For equally spaced positions:

$$
\mathbf P_i=(X_i,Y_i,Z_i),\qquad
\mathbf V_i\approx\frac{\mathbf P_{i+1}-\mathbf P_{i-1}}{2\Delta t}
$$

Positions in millimetres and time in seconds give velocity in mm/s. These labels represent motion relative to the chosen camera frame; they are not automatically robot-base or world-frame velocities. With a moving camera, changes in camera-relative coordinates include camera motion.

Stereo-derived labels have annotation and calibration uncertainty. Reject unreliable correspondences, mark missing values explicitly and inspect reconstruction noise before differentiating positions. Keep labels derived from dataset annotations separate from predictions used for evaluation.

### 3. Build the training dataset

Create a table linking each video sequence and instrument track to its tip and velocity labels.

| Field | Purpose |
| --- | --- |
| Sequence ID, frame index and timestamp | Locate frames and preserve acquisition timing. |
| Instrument ID and tip definition | Identify the same instrument point across time. |
| Tip pixel coordinates and visibility | Provide localization labels and valid-sample masks. |
| Camera-frame position | Store reconstructed positions with verified units. |
| Velocity components | Supply the physical regression targets. |
| Validity/quality flags | Exclude missing or unreliable labels. |

A PyTorch data loader uses this table to load short video clips and their target velocities. **Velocity labels are training targets, not input features.** Split by whole sequences before constructing overlapping clips to avoid leakage.

### 4. Train a temporal velocity model

Initially keep YOLO as the localizer and train a separate velocity predictor. A proposed baseline is:

**RGB clip + tracked tip coordinates/confidence + timing → image encoder → temporal model (for example, a GRU) → velocity regression head.**

The output could be camera-frame horizontal and vertical velocity, or all three components. The output dimensionality must match the chosen labels. This architecture is proposed, not yet implemented or validated.

A regression loss compares predicted velocities with reference labels. A single-frame detection head alone does not explicitly observe movement, so the velocity model needs temporal information. If crops are moved or resized, retain original coordinates and crop/scale metadata to preserve motion and scale cues.

### 5. Explore optical flow

Optical flow is an optional additional input describing apparent pixel displacement between frames. A model such as [RAFT](https://arxiv.org/abs/2003.12039) could provide motion features around the tracked instrument.

Optical flow does not directly provide mm/s or resolve physical scale. Compare the baseline against a version with flow using the same held-out sequences and velocity targets.

### 6. Annotate previously unlabelled video

At inference, load the video frames and timing, detect and track tips, and predict velocities with the trained temporal model. Export per-instrument predictions and display tip locations, IDs and velocity estimates on the video.

For monocular deployment, the predictor should use only one camera viewpoint and other information available at deployment. Stereo data can supply training supervision without being required at inference.

## Evaluation and limitations

- Evaluate tip-localization error and velocity error separately on held-out labelled sequences. Report missing predictions and coverage alongside error metrics.
- Evaluate the complete pipeline using predicted detections, not only ideal annotated tip locations.
- A qualitative overlay on unlabelled video demonstrates application, not measured physical velocity accuracy.
- Monocular image motion alone does not uniquely determine physical scale. Learned mm/s prediction depends on visual cues and training assumptions; transfer across camera geometry, working distance, instruments and scenes must be tested.
- Reflections, occlusion, detection jitter, identity switches and stereo mismatch can corrupt trajectories and velocity estimates.
- Verify acquisition timing before reporting physical velocity. Do not confuse playback timing with capture timing.
- Keep camera-relative velocity distinct from robot-base or world-relative motion.

## Existing image-plane velocity baseline

The current tracking baseline operates on pixel coordinates `p = (u, v)`, with the image origin at the upper-left, `u` increasing rightward and `v` downward.

For consecutive matched detections, it calculates displacement divided by elapsed time. For uniformly sampled video whose FPS accurately represents capture timing, this is:

$$
\mathbf v_{\mathrm{image},i}
=\frac{\mathbf p_i-\mathbf p_{i-1}}{\Delta t}
$$

The result is in pixels/second. Speed is the magnitude of that velocity vector. The tracker omits velocity for new detections, unresolved matches and the first frame after a gap. This baseline supports tracking analysis but is distinct from the planned learned physical velocity predictor.

## Intended visualization

<table>
  <tr>
    <td width="50%" align="center">
      <img src="./assets/fig1.png" width="100%" alt="Surgical instrument with a cyan bounding box, a tip marker, and an overlay showing position and velocity components.">
      <br>
      <strong>Figure 1. Single-frame instrument annotation.</strong>
      A cyan cross marks the selected tip location, while a bounding box indicates the instrument region. The overlay displays tip coordinates and image-plane velocity components.
    </td>
    <td width="50%" align="center">
      <img src="./assets/fig2.png" width="100%" alt="Instrument motion illustrated at three timestamps, with a trajectory overlay and plots of tip position and speed over time.">
      <br>
      <strong>Figure 2. Tool-tip tracking and kinematic visualization.</strong>
      Overlaid instrument positions at 0.0, 0.1, and 0.2 seconds illustrate motion over time. The lower panels show example position and speed curves.
    </td>
  </tr>
</table>

These figures illustrate the intended annotation and visualization concepts; the displayed values and curves should not be treated as validated measurements. 


## Longer-term direction

Explore surgical gesture recognition using tip trajectories and velocities together with broader image-motion patterns. This is a downstream research direction; the primary goal remains kinematics labelling of video datasets.

## Course information

- click [here](https://whiteboard.cloud.microsoft/me/whiteboards/p/c3BvOmh0dHBzOi8vbGl2ZWpvaG5zaG9wa2lucy1teS5zaGFyZXBvaW50LmNvbS9wZXJzb25hbC9hY2hlcnV2Ml9qaF9lZHU%3d/b!reAG3KHhS0aF67rzwluLoFWuxjk4lm5Gk9qaOkSgffEI5GWe4kiDT5oXWVzVnAmd/01DEIPJTX726LX3CGVOREZ3MOCWX6DKZ3L?fromShare=true) to write board
- click [here](https://jhu.instructure.com/courses/132294) to Canvas
## Group members
- Alex Luna-Ow <alunaow1@jh.edu>
- Ahmed Cheruvattam <acheruv2@jh.edu>
- Vincentz Luo <zluo46@jh.edu>
- Will Luo <yluo87@jh.edu>
