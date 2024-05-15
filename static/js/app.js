// const webcam = document.getElementById("camera");

// if (navigator.mediaDevices.getUserMedia) {
// 	navigator.mediaDevices
// 		.getUserMedia({ video: true })
// 		.then((stream) => {
// 			webcam.srcObject = stream;
// 		})
// 		.catch((error) => {
// 			console.error("Error accessing webcam:", error);
// 		});
// } else {
// 	console.error("getUserMedia not supported in this browser.");
// }

const canvas = document.getElementById("gameCanvas");
const canvasWidth = parseInt(canvas.style.width.replace("px", ""));
const canvasHeight = parseInt(canvas.style.height.replace("px", ""));
const ctx = canvas.getContext("2d");
let playerRacket = {
	x1: 100,
	y1: 100,
	x2: 300,
	y2: 300,
	width: 10, // Thickness of player racket
};

const opponentRacket = {
	x1: 900,
	y1: 20,
	x2: 900,
	y2: 480,
	width: 10, // Thickness of opponent racket
};
const ball = {
	x: canvas.width / 2,
	y: canvas.height / 2,
	radius: 10,
	speedX: 3,
	speedY: 3,
};

function drawRacket(racket) {
	// Set the outline properties
	ctx.lineWidth = racket.width + 4; // Adjust the width for the outline
	ctx.strokeStyle = "black"; // Color for the outline
	ctx.lineJoin = "round"; // To make corners smooth
	ctx.lineCap = "round"; // To make the ends of lines smooth

	// Draw the outline
	ctx.beginPath();
	ctx.moveTo(racket.x1, racket.y1);
	ctx.lineTo(racket.x2, racket.y2);
	ctx.stroke();

	// Draw the main stroke
	ctx.lineWidth = racket.width;
	ctx.strokeStyle = "white"; // Color for the main stroke
	ctx.stroke();
}

// Draw Rackets
function drawRackets() {
	// Draw player racket
	drawRacket(playerRacket);

	// Draw opponent racket
	drawRacket(opponentRacket);
}

// Ball collision with rackets
function checkBallRacketCollision(racket) {
	// Calculate the center of the ball
	const ballCenterX = ball.x;
	const ballCenterY = ball.y;

	// Calculate the direction of the racket
	const racketDirectionX = racket.x2 - racket.x1;
	const racketDirectionY = racket.y2 - racket.y1;

	// Calculate the vector from the racket's starting point to the ball's center
	const toBallX = ballCenterX - racket.x1;
	const toBallY = ballCenterY - racket.y1;

	// Calculate the dot product of the racket's direction and the vector to the ball
	const dotProduct = toBallX * racketDirectionX + toBallY * racketDirectionY;

	// Calculate the magnitude of the racket's direction squared
	const magnitudeSquared =
		racketDirectionX * racketDirectionX + racketDirectionY * racketDirectionY;

	// Calculate the projection of the vector to the ball onto the racket's direction
	const projection = dotProduct / magnitudeSquared;

	// Calculate the closest point on the racket to the ball
	let closestX, closestY;
	if (projection < 0) {
		closestX = racket.x1;
		closestY = racket.y1;
	} else if (projection > 1) {
		closestX = racket.x2;
		closestY = racket.y2;
	} else {
		closestX = racket.x1 + projection * racketDirectionX;
		closestY = racket.y1 + projection * racketDirectionY;
	}

	// Calculate the distance between the closest point on the racket and the ball's center
	const distance = Math.sqrt(
		(ballCenterX - closestX) ** 2 + (ballCenterY - closestY) ** 2
	);

	// Check if the distance is less than the sum of the ball's radius and half the racket's thickness
	if (distance < ball.radius + racket.width / 2) {
		// Calculate the collision normal
		const normalX = ballCenterX - closestX;
		const normalY = ballCenterY - closestY;
		const normalLength = Math.sqrt(normalX * normalX + normalY * normalY);
		const collisionNormalX = normalX / normalLength;
		const collisionNormalY = normalY / normalLength;

		// Move the ball out of the racket
		ball.x = closestX + collisionNormalX * (ball.radius + racket.width / 2);
		ball.y = closestY + collisionNormalY * (ball.radius + racket.width / 2);

		return true;
	}
	return false;
}

// Update Game
function update() {
	// Update ball position
	ball.x += ball.speedX;
	ball.y += ball.speedY;

	// Ball collision with walls
	if (
		ball.y + ball.speedY > canvas.height - ball.radius ||
		ball.y + ball.speedY < ball.radius
	) {
		ball.speedY = -ball.speedY;
	}

	if (
		ball.x + ball.speedX > canvas.width - ball.radius ||
		ball.x + ball.speedX < ball.radius
	) {
		ball.speedX = -ball.speedX;
	}

	// Ball collision with player racket
	if (checkBallRacketCollision(playerRacket)) {
		ball.speedX = -ball.speedX;
	}

	// Ball collision with opponent racket
	if (checkBallRacketCollision(opponentRacket)) {
		ball.speedX = -ball.speedX;
	}
}

// Draw everything
function draw() {
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	drawRackets();
	drawBall();
}

// Draw Ball
function drawBall() {
	ctx.beginPath();
	ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
	ctx.fillStyle = "red";
	ctx.fill();
}
let direction = 0;
function simulatePlayerPosition() {
	// Simulate decrease in player position
	if (playerRacket.y2 >= 490) {
		direction = 0;
	} else if (playerRacket.y1 <= 10) {
		direction = 1;
	}
	if (direction == 0) {
		playerRacket.y1 -= 1; // Decrease y-coordinate of the top point of the player racket
		playerRacket.y2 -= 1; // Decrease y-coordinate of the bottom point of the player racket
		playerRacket.x1 -= 1;
		playerRacket.x2 += 1;
	}
	if (direction == 1) {
		playerRacket.y1 += 1; // Decrease y-coordinate of the top point of the player racket
		playerRacket.y2 += 1; // Decrease y-coordinate of the bottom point of the player racket
		playerRacket.x1 += 1;
		playerRacket.x2 -= 1;
	}
}
// Game Loop
function gameLoop() {
	//simulatePlayerPosition();
	update();
	draw();
	requestAnimationFrame(gameLoop);
}

var socket = io.connect();

//receive details from server
socket.on("updateFingersPositions", function (msg) {
	//assign racket position
	playerRacket.x1 = msg["x1"] * 1000;
	playerRacket.y1 = msg["y1"] * 500;
	playerRacket.x2 = msg["x2"] * 1000;
	playerRacket.y2 = msg["y2"] * 500;
	console.log(msg);
});

// Start the game loop
gameLoop();
