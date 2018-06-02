import { Component, OnInit, AfterViewInit } from '@angular/core';
import * as posenet from '@tensorflow-models/posenet';
import * as mobilenet from '@tensorflow-models/mobilenet';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'ng-tensorflow';

  loadingPosenet: boolean = true;
  loadingMobilenet: boolean = true;
  predictions: any;
  net: posenet.PoseNet;
  model: mobilenet.MobileNet;
  img: any;

  ngOnInit() {
    let elem = this;

    async function loadPosenet() {
      // load the posenet model from a checkpoint
      elem.net = await posenet.load();
      elem.loadingPosenet = false;
    }
    loadPosenet();

    async function loadMobilenet() {
      // load the posenet model from a checkpoint
      elem.model = await mobilenet.load();
      elem.loadingMobilenet = false
    }
    loadMobilenet();

  }

  handleInputChange(fileInput) {
    this.img = fileInput.target.files[0];

    let reader = new FileReader();

    reader.onload = (e: any) => {
      this.img = e.target.result;

      this.onImgChange();
    }

    reader.readAsDataURL(fileInput.target.files[0]);
  }

  onImgChange() {
    let elem = this;

    document.getElementById("canvasDiv").innerHTML = "";

    const imageElement: any = document.getElementById('test');

    // get heigth and width of img
    imageElement.addEventListener('load', function () {
      var cnvsDiv = document.getElementById("canvasDiv");

      var canvas = document.createElement('canvas');

      canvas.style.position = "absolute";
      canvas.style.left = this.offsetLeft + "px";
      canvas.style.top = this.offsetTop + "px";
      canvas.id = "canvasLabel";
      canvas.width = this.naturalWidth;
      canvas.height = this.naturalHeight;

      cnvsDiv.appendChild(canvas);

      var cnvsCtx = canvas.getContext("2d");
      cnvsCtx.fillStyle = "red";
      cnvsCtx.font = "15px Verdana";

      const imageScaleFactor = 0.5;
      const outputStride = 16;
      const flipHorizontal = false;

      async function estimatePoseOnImage(imageElement) {
        const pose = await elem.net.estimateSinglePose(imageElement, imageScaleFactor, flipHorizontal, outputStride);

        return pose;
      }

      async function predictImage(imageElement) {
        // Classify the image.
        elem.predictions = await elem.model.classify(imageElement);
      }

      predictImage(imageElement).then(p => elem.loadingMobilenet = false, err => console.error(err));

      const pose = estimatePoseOnImage(imageElement);

      function drawPoint(x, y, part, cnvsCtx) {
        cnvsCtx.fillText(part, x, y);
        // cnvsCtx.beginPath();
        // cnvsCtx.fillRect(x, y, 2, 2);
        // cnvsCtx.stroke();
      }

      function drawLine(posA, posB, ctx) {
        ctx.beginPath();
        ctx.strokeStyle="#FF0000";
        ctx.moveTo(posA.x, posA.y);
        ctx.lineTo(posB.x, posB.y);
        ctx.stroke();
      }

      pose.then(joints => {
        console.log("joints");
        console.log(joints);

        for (let joint of joints.keypoints) {
          drawPoint(joint.position.x, joint.position.y, joint.part, cnvsCtx);
        }

        // UPPER
        drawLine(joints.keypoints[5].position, joints.keypoints[6].position, cnvsCtx);
        drawLine(joints.keypoints[5].position, joints.keypoints[7].position, cnvsCtx);
        drawLine(joints.keypoints[9].position, joints.keypoints[7].position, cnvsCtx);
        drawLine(joints.keypoints[6].position, joints.keypoints[8].position, cnvsCtx);
        drawLine(joints.keypoints[10].position, joints.keypoints[8].position, cnvsCtx);
        // LOWER
        drawLine(joints.keypoints[11].position, joints.keypoints[12].position, cnvsCtx);
        drawLine(joints.keypoints[11].position, joints.keypoints[13].position, cnvsCtx);
        drawLine(joints.keypoints[15].position, joints.keypoints[13].position, cnvsCtx);
        drawLine(joints.keypoints[12].position, joints.keypoints[14].position, cnvsCtx);
        drawLine(joints.keypoints[16].position, joints.keypoints[14].position, cnvsCtx);

      }, err => console.error(err));
    });
  }

}
