// handling audio recording on browser

let mediaRecorder;
let chunks = [];

document.getElementById('start') = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
  mediaRecorder.start();
};

document.getElementById('stop') = async () => {
  mediaRecorder.stop();
  const audioBlob = new Blob(chunks, { type: 'audio/wav' });
  const formData = new FormData();
  formData.append('audio', audioBlob, 'audio.wav');

  await fetch('/record', {method: "POST", body: formData});

};
