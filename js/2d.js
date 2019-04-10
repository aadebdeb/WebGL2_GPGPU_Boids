(function() {

const FILL_SCREEN_VERTEX_SHADER_SOURCE =
`#version 300 es

layout (location = 0) in vec2 position;

void main(void) {
  gl_Position = vec4(position, 0.0, 1.0);
}
`

  const INITIALIZE_BOID_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

layout (location = 0) out vec2 o_position;
layout (location = 1) out vec2 o_velocity;

uniform vec2 u_randomSeed;
uniform uint u_boidNum;
uniform uint u_boidTextureSize;
uniform float u_maxSpeed;

uint convertCoordToIndex(uvec2 coord, uint sizeX) {
  return coord.x + sizeX * coord.y;
}

float random(vec2 x){
  return fract(sin(dot(x,vec2(12.9898, 78.233))) * 43758.5453);
}

void main(void) {
  uint index = convertCoordToIndex(uvec2(gl_FragCoord.xy), u_boidTextureSize);

  if (index >= u_boidNum) { // unused pixels
    o_position = vec2(0.0);
    o_velocity = vec2(0.0);
    return;
  }

  o_position = vec2(
    random(gl_FragCoord.xy * 0.013 + u_randomSeed + vec2(32.19, 27.51)),
    random(gl_FragCoord.xy * 0.029 + u_randomSeed + vec2(19.56, 11.34))
  );
  o_velocity = normalize(vec2(
    random(gl_FragCoord.xy * 0.059 + u_randomSeed + vec2(27.31, 16.91)),
    random(gl_FragCoord.xy * 0.038 + u_randomSeed + vec2(25.95, 19.47))
  ) * 2.0 - 1.0) * u_maxSpeed;
}
`

  const UPDATE_BOID_FRAGMENT_SHADER_SOURCE = 
`#version 300 es

precision highp float;

layout (location = 0) out vec2 o_position;
layout (location = 1) out vec2 o_velocity;

uniform sampler2D u_positionTexture;
uniform sampler2D u_velocityTexture;
uniform sampler2D u_forceTexture;
uniform float u_deltaTime;
uniform float u_maxSpeed;
uniform uint u_boidNum;
uniform uint u_boidTextureSize;

uint convertCoordToIndex(uvec2 coord, uint sizeX) {
  return coord.x + sizeX * coord.y;
}

vec2 limit(vec2 v, float max) {
  if (length(v) < max) {
    return normalize(v) * max;
  }
  return v;
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);

  uint index = convertCoordToIndex(uvec2(coord), u_boidTextureSize);
  if (index >= u_boidNum) { // unused pixels
    o_position = vec2(0.0);
    o_velocity = vec2(0.0);
    return;
  }

  vec2 position = texelFetch(u_positionTexture, coord, 0).xy;
  vec2 velocity = texelFetch(u_velocityTexture, coord, 0).xy;
  vec2 force = texelFetch(u_forceTexture, coord, 0).xy;

  vec2 nextVelocity = limit(velocity + u_deltaTime * force, u_maxSpeed);
  vec2 nextPosition = position + u_deltaTime * velocity;

  if (nextPosition.x < 0.0) {
    nextPosition.x += 1.0;
  }
  if (nextPosition.x > 1.0) {
    nextPosition.x -= 1.0;
  }
  if (nextPosition.y < 0.0) {
    nextPosition.y += 1.0;
  }
  if (nextPosition.y > 1.0) {
    nextPosition.y -= 1.0;
  }

  o_position = nextPosition;
  o_velocity = nextVelocity;
}
`

const COMPUTE_FORCE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;
precision highp isampler2D;
precision highp usampler2D;

out vec2 o_force;

uniform sampler2D u_positionTexture;
uniform sampler2D u_velocityTexture;
uniform usampler2D u_bucketTexture;
uniform isampler2D u_bucketReferrerTexture;

uniform vec3 u_radius; // x: separation, y: alignment: z: cohesion
uniform vec3 u_weight; // x: separation, y: alignment: z: cohesion
uniform float u_maxSpeed;
uniform float u_maxForce;

uniform uint u_boidNum;
uniform uint u_boidTextureSize;

uniform float u_bucketSize;

float simulationSpace = 1.0;

uint convertCoordToIndex(uvec2 coord, uint sizeX) {
  return coord.x + sizeX * coord.y;
}

ivec2 convertIndexToCoord(int index, int sizeX) {
  return ivec2(index % sizeX, index / sizeX);
}

void findNeighbors(vec2 position, ivec2 bucketPosition, ivec2 bucketNum, int particleTextureSizeX, int bucketReferrerTextureSizeX, inout vec3 separation, inout vec3 alignment, inout vec3 cohesion) {
  if (bucketPosition.x < 0 || bucketPosition.x >= bucketNum.x || bucketPosition.y < 0 || bucketPosition.y >= bucketNum.y) {
    return;
  }
  int bucketIndex = bucketPosition.x + bucketNum.x * bucketPosition.y;
  ivec2 coord = convertIndexToCoord(bucketIndex, bucketReferrerTextureSizeX);

  ivec2 bucketReferrer = texelFetch(u_bucketReferrerTexture, coord, 0).xy;

  if (bucketReferrer.x == -1 || bucketReferrer.y == -1) {
    return;
  }

  for (int i = bucketReferrer.x; i <= bucketReferrer.y; i++) {
    uvec2 bucket = texelFetch(u_bucketTexture, convertIndexToCoord(i, particleTextureSizeX), 0).xy;

    int particleIndex = int(bucket.y);

    ivec2 particleCoord = convertIndexToCoord(particleIndex, particleTextureSizeX);
    vec2 otherPos = texelFetch(u_positionTexture, particleCoord, 0).xy;
    vec2 otherVel = texelFetch(u_velocityTexture, particleCoord, 0).xy;

    float dist = length(position - otherPos);

    if (dist == 0.0) {
      continue;
    }

    if (dist < u_radius.x) {
      separation.xy += normalize(position - otherPos) / dist;
      separation.z += 1.0;
    }

    if (dist < u_radius.y) {
      alignment.xy += otherVel;
      alignment.z += 1.0;
    }

    if (dist < u_radius.z) {
      cohesion.xy += otherPos;
      cohesion.z += 1.0;
    }
  }
}

vec2 limit(vec2 v, float max) {
  if (length(v) > max) {
    return normalize(v) * max;
  }
  return v;
}

vec2 computeForce() {

  ivec2 coord = ivec2(gl_FragCoord.xy);
  vec2 position = texelFetch(u_positionTexture, coord, 0).xy;
  vec2 velocity = texelFetch(u_velocityTexture, coord, 0).xy;

  int particleTextureSizeX = textureSize(u_positionTexture, 0).x;
  int bucketReferrerTextureSizeX = textureSize(u_bucketReferrerTexture, 0).x;

  vec2 bucketPosition = position / u_bucketSize;
  int xOffset = fract(bucketPosition.x) < 0.5 ? -1 : 1;
  int yOffset = fract(bucketPosition.y) < 0.5 ? -1 : 1;

  ivec2 bucketPosition00 = ivec2(bucketPosition);
  ivec2 bucketPosition10 = bucketPosition00 + ivec2(xOffset, 0);
  ivec2 bucketPosition01 = bucketPosition00 + ivec2(0, yOffset);
  ivec2 bucketPosition11 = bucketPosition00 + ivec2(xOffset, yOffset);

  ivec2 bucketNum = ivec2(simulationSpace / u_bucketSize) + 1;

  vec3 separation = vec3(0.0);
  vec3 alignment = vec3(0.0);
  vec3 cohesion = vec3(0.0);
  findNeighbors(position, bucketPosition00, bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, separation, alignment, cohesion);
  findNeighbors(position, bucketPosition10, bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, separation, alignment, cohesion);
  findNeighbors(position, bucketPosition01, bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, separation, alignment, cohesion);
  findNeighbors(position, bucketPosition11, bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, separation, alignment, cohesion);

  vec2 separationForce = vec2(0.0);
  if (separation.z != 0.0) {
    vec2 desiredVelocity = normalize(separation.xy) * u_maxSpeed;
    separationForce = limit(desiredVelocity - velocity, u_maxForce);
  }

  vec2 alignmentForce = vec2(0.0);
  if (alignment.z != 0.0) {
    vec2 desiredVelocity = normalize(alignment.xy) * u_maxSpeed;
    alignmentForce = limit(desiredVelocity - velocity, u_maxForce);
  }

  vec2 cohesionForce = vec2(0.0);
  if (cohesion.z != 0.0) {
    vec2 target = cohesion.xy / cohesion.z;
    vec2 desiredVelocity = normalize(target - position) * u_maxSpeed;
    cohesionForce = limit(desiredVelocity - velocity, u_maxForce);
  }

  separationForce *= u_weight.x;
  alignmentForce *= u_weight.y;
  cohesionForce *= u_weight.z;

  return separationForce + alignmentForce + cohesionForce;
}

void main(void) {
  uint index = convertCoordToIndex(uvec2(gl_FragCoord.xy), u_boidTextureSize);
  if (index >= u_boidNum) { // unused pixels
    o_force = vec2(0.0);
    return;
  }
  o_force = computeForce();
}
`;

  const RENDER_BOID_VERTEX_SHADER_SOURCE =
`#version 300 es

precision highp isampler2D;
precision highp usampler2D;

out float v_angle;

uniform sampler2D u_positionTexture;
uniform sampler2D u_velocityTexture;
uniform vec2 u_canvasSize;
uniform float u_boidSize;

float simulationSpace = 1.0;

ivec2 convertIndexToCoord(int index, int sizeX) {
  return ivec2(index % sizeX, index / sizeX);
}

void main(void) {
  vec2 scale = min(u_canvasSize.x, u_canvasSize.y) / u_canvasSize;
  ivec2 coord = convertIndexToCoord(gl_VertexID, textureSize(u_positionTexture, 0).x);
  vec2 position = texelFetch(u_positionTexture, coord, 0).xy;
  vec2 velocity = texelFetch(u_velocityTexture, coord, 0).xy;
  if (velocity.x == 0.0) {
    v_angle = 3.14 * 0.25;
  } else {
    v_angle = atan(velocity.y, velocity.x);
  }
  gl_Position = vec4(scale * (position * 2.0 - 1.0), 0.0, 1.0);
  gl_PointSize = u_boidSize;
}
`

  const RENDER_BOID_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in float v_angle;

out vec4 o_color;

mat2 rotate(float r) {
  float c = cos(r);
  float s = sin(r);
  return mat2(c, s, -s, c);
}

float sdTriangleIsosceles(vec2 p, vec2 q) {
  p.x = abs(p.x);
  vec2 a = p - q * clamp(dot(p, q) / dot(q, q), 0.0, 1.0);
  vec2 b = p - q * vec2(clamp(p.x / q.x, 0.0, 1.0), 1.0);
  float s = -sign(q.y);
  vec2 d = min(vec2(dot(a, a), s * (p.x * q.y - p.y * q.x)), vec2(dot(b, b), s * (p.y - q.y)));
  return -sqrt(d.x) * sign(d.y);
}

void main(void) {
  vec2 p = ((gl_PointCoord - 0.5) * 2.0).yx;
  p *= rotate(v_angle);
  p.y *= -1.0;
  if (sdTriangleIsosceles(p * 1.25, vec2(1.0, 5.0)) <= 0.0) {
    o_color = vec4(vec3(0.8), 1.0);
  } else {
    discard;
  }
}
`

const INITIALIZE_BUCKET_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out uvec2 o_bucket;

uniform sampler2D u_positionTexture;
uniform float u_neighborRadius;
uniform uint u_boidNum;

float simulationSpace = 1.0;

uint convertCoordToIndex(uvec2 coord, uint sizeX) {
  return coord.x + sizeX * coord.y;
}

uint getBucketIndex(vec2 position) {
  uvec2 bucketCoord = uvec2(position / (2.0 * u_neighborRadius));
  uvec2 bucketNum = uvec2(simulationSpace / (2.0 * u_neighborRadius)) + 1u;
  return bucketCoord.x + bucketCoord.y * bucketNum.x;
}

void main(void) {
  uint positionTextureSizeX = uint(textureSize(u_positionTexture, 0).x);
  uint boidIndex = convertCoordToIndex(uvec2(gl_FragCoord.xy), positionTextureSizeX);
  if (boidIndex < u_boidNum) {
    vec2 position = texelFetch(u_positionTexture, ivec2(gl_FragCoord.xy), 0).xy;
    uint bucketIndex = getBucketIndex(position);
    o_bucket = uvec2(bucketIndex, boidIndex);
  } else {
    o_bucket = uvec2(65530, 65530);
  }
}
`

  const SWAP_BUCKET_INDEX_FRAGMENT_SHADER_SOURCE = 
`#version 300 es

precision highp float;
precision highp usampler2D;

out uvec2 o_bucket;

uniform usampler2D u_bucketTexture;
uniform uint u_size;
uniform uint u_blockStep;
uniform uint u_subBlockStep;

uint convertCoordToIndex(uvec2 coord, uint sizeX) {
  return coord.x + sizeX * coord.y;
}

uvec2 convertIndexToCoord(uint index, uint sizeX) {
  return uvec2(index % sizeX, index / sizeX);
}

void main(void) {
  uint index = convertCoordToIndex(uvec2(gl_FragCoord.xy), u_size);
  uint d = 1u << (u_blockStep - u_subBlockStep);

  bool up = ((index >> u_blockStep) & 2u) == 0u;

  uint targetIndex;
  bool first = (index & d) == 0u;
  if (first) {
    targetIndex = index | d;
  } else {
    targetIndex = index & ~d;
    up = !up;
  }

  uvec2 a = texelFetch(u_bucketTexture, ivec2(gl_FragCoord.xy), 0).xy;
  uvec2 b = texelFetch(u_bucketTexture, ivec2(convertIndexToCoord(targetIndex, u_size)), 0).xy;

  if (a.x == b.x || (a.x >= b.x) == up) {
    o_bucket = b;
  } else {
    o_bucket = a;
  }
}

`;

  const INITIALIZE_BUCKET_REFERRER_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;
precision highp usampler2D;

out ivec2 o_referrer;

uniform ivec2 u_bucketReferrerTextureSize;
uniform usampler2D u_bucketTexture;
uniform float u_neighborRadius;
uniform int u_boidNumN;

float simulationSpace = 1.0;

int convertCoordToIndex(ivec2 coord, int sizeX) {
  return coord.x + sizeX * coord.y;
}

ivec2 convertIndexToCoord(int index, int sizeX) {
  return ivec2(index % sizeX, index / sizeX);
}

int getBucketIndex(int particleIndex, int particleTextureSizeX) {
  return int(texelFetch(u_bucketTexture, ivec2(convertIndexToCoord(particleIndex, particleTextureSizeX)), 0).x);
}


int binarySearchMinIndex(int target, int from, int to, int particleTextureSizeX) {
  for (int i = 0; i < u_boidNumN + 1; i++) {
    int middle = from + (to - from) / 2;
    int bucketIndex = getBucketIndex(middle, particleTextureSizeX);
    if (bucketIndex < target) {
      from = middle + 1;
    } else {
      to = middle;
    }
    if (from == to) {
      if (getBucketIndex(from, particleTextureSizeX) == target) {
        return from;
      } else {
        return -1;
      }
    }
  }
  return -1;
}

int binarySearchMaxIndex(int target, int from, int to, int particleTextureSizeX) {
  for (int i = 0; i < u_boidNumN + 1; i++) {
    int middle = from + (to - from) / 2 + 1;
    int bucketIndex = getBucketIndex(middle, particleTextureSizeX);
    if (bucketIndex > target) {
      to = middle - 1;
    } else {
      from = middle;
    }
    if (from == to) {
      if (getBucketIndex(from, particleTextureSizeX) == target) {
        return from;
      } else {
        return -1;
      }
    }
  }
  return -1;
}

ivec2 binarySearchRange(int target, int from, int to) {
  int particleTextureSizeX = textureSize(u_bucketTexture, 0).x;
  from =  binarySearchMinIndex(target, from, to, particleTextureSizeX);
  to = from == -1 ? -1 : binarySearchMaxIndex(target, from, to, particleTextureSizeX);
  return ivec2(from, to);
}

void main(void) {
  int bucketIndex = convertCoordToIndex(ivec2(gl_FragCoord.xy), u_bucketReferrerTextureSize.x);
  ivec2 bucketNum = ivec2(simulationSpace / (2.0 * u_neighborRadius)) + 1;
  int maxBucketIndex = bucketNum.x * bucketNum.y;

  if (bucketIndex >= maxBucketIndex) {
    o_referrer = ivec2(-1, -1);
    return;
  }

  ivec2 particleTextureSize = textureSize(u_bucketTexture, 0);
  int particleNum = particleTextureSize.x * particleTextureSize.y;

  o_referrer = binarySearchRange(bucketIndex, 0, particleNum - 1);
}
`

  const VERTICES_POSITION = new Float32Array([
    -1.0, -1.0,
    1.0, -1.0,
    -1.0,  1.0,
    1.0,  1.0
  ]);

  const VERTICES_INDEX = new Int16Array([
    0, 1, 2,
    3, 2, 1
  ]);


  function createInitializeBoidProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, INITIALIZE_BOID_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createUpdateBoidProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, UPDATE_BOID_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createComputeForceProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, COMPUTE_FORCE_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createRenderBoidProgram(gl) {
    const vertexShader = createShader(gl, RENDER_BOID_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, RENDER_BOID_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createInitializeBucketProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, INITIALIZE_BUCKET_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createSwapBucketIndexProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, SWAP_BUCKET_INDEX_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createInitializeBucketReferrerProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, INITIALIZE_BUCKET_REFERRER_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createBoidFramebuffer(gl, size) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const positionTexture = createTexture(gl, size, gl.RG32F, gl.RG, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, positionTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, positionTexture, 0);
    const velocityTexture = createTexture(gl, size, gl.RG32F, gl.RG, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, velocityTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, velocityTexture, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      positionTexture: positionTexture,
      velocityTexture: velocityTexture
    };
  }

  function createForceFramebuffer(gl, size) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const forceTexture = createTexture(gl, size, gl.RG32F, gl.RG, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, forceTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, forceTexture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      forceTexture: forceTexture
    };
  }

  function createBucketFramebuffer(gl, size) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const bucketTexture = createTexture(gl, size, gl.RG32UI, gl.RG_INTEGER, gl.UNSIGNED_INT);
    gl.bindTexture(gl.TEXTURE_2D, bucketTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, bucketTexture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      bucketTexture: bucketTexture
    };
  }

  function createBucketReferrerFramebuffer(gl, size) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const bucketReferrerTexture = createTexture(gl, size, gl.RG32I, gl.RG_INTEGER, gl.INT);
    gl.bindTexture(gl.TEXTURE_2D, bucketReferrerTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, bucketReferrerTexture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      bucketReferrerTexture: bucketReferrerTexture
    };
  }

  const canvas = document.getElementById('canvas');
  const resizeCanvas = function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  };
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  const gl = canvas.getContext('webgl2');
  gl.getExtension('EXT_color_buffer_float');

  const initializeBoidProgram = createInitializeBoidProgram(gl);
  const updateBoidProgram = createUpdateBoidProgram(gl);
  const computeForceProgram = createComputeForceProgram(gl);
  const renderBoidProgram = createRenderBoidProgram(gl);
  const initializeBucketProgram = createInitializeBucketProgram(gl);
  const swapBucketIndexProgram = createSwapBucketIndexProgram(gl);
  const initializeBucketReferrerProgram = createInitializeBucketReferrerProgram(gl);

  const initializeBoidUniforms = getUniformLocations(gl, initializeBoidProgram, ['u_randomSeed', 'u_boidNum', 'u_boidTextureSize', 'u_maxSpeed']);
  const updateBoidUniforms = getUniformLocations(gl, updateBoidProgram, ['u_positionTexture', 'u_velocityTexture', 'u_forceTexture', 'u_deltaTime', 'u_boidNum', 'u_boidTextureSize', 'u_maxSpeed']);
  const computeForceUniforms = getUniformLocations(gl, computeForceProgram,
    ['u_positionTexture', 'u_velocityTexture', 'u_bucketTexture', 'u_bucketReferrerTexture', 'u_boidNum', 'u_boidTextureSize', 'u_radius', 'u_weight', 'u_maxSpeed', 'u_maxForce', 'u_bucketSize']);
  const renderBoidUniforms = getUniformLocations(gl, renderBoidProgram, ['u_positionTexture', 'u_velocityTexture', 'u_canvasSize', 'u_boidSize']);
  const initializeBucketUniforms = getUniformLocations(gl, initializeBucketProgram, ['u_positionTexture', 'u_neighborRadius', 'u_boidNum']);
  const swapBucketIndexUniforms = getUniformLocations(gl, swapBucketIndexProgram, ['u_bucketTexture', 'u_size', 'u_blockStep', 'u_subBlockStep']);
  const initializeBucketReferrerUniforms = getUniformLocations(gl, initializeBucketReferrerProgram, ['u_bucketTexture', 'u_neighborRadius', 'u_bucketReferrerTextureSize', 'u_boidNumN']);

  const fillScreenVao = createVao(gl,
    [{buffer: createVbo(gl, VERTICES_POSITION), size: 2, index: 0}],
    createIbo(gl, VERTICES_INDEX)
  );

  const stats = new Stats();
  document.body.appendChild(stats.dom);

  const gui = new dat.GUI();
  const parameters = {
    dynamic: {
      'separation weight': 1.0,
      'alignment weight': 1.5,
      'cohesion weight': 1.5,
      'max speed': 0.05,
      'max force': 0.05,
      'boid size': 20.0,
    },
    static: {
      'boid num': 1024,
      'separation radius': 0.025,
      'alignment radius': 0.051,
      'cohesion radius': 0.051,
    },
    'reset': () => reset()
  };
  const dynamicFolder = gui.addFolder('dynamic parameters');
  dynamicFolder.add(parameters.dynamic, 'separation weight', 0.0, 2.0);
  dynamicFolder.add(parameters.dynamic, 'alignment weight', 0.0, 2.0);
  dynamicFolder.add(parameters.dynamic, 'cohesion weight', 0.0, 2.0);
  dynamicFolder.add(parameters.dynamic, 'max speed', 0.0, 1.0);
  dynamicFolder.add(parameters.dynamic, 'max force', 0.0, 1.0);
  dynamicFolder.add(parameters.dynamic, 'boid size', 1.0, 50.0);
  const staticFolder = gui.addFolder('static parameters');
  staticFolder.add(parameters.static, 'boid num', 1, 65536);
  staticFolder.add(parameters.static, 'separation radius', 0.05, 0.1);
  staticFolder.add(parameters.static, 'alignment radius', 0.05, 0.1);
  staticFolder.add(parameters.static, 'cohesion radius', 0.05, 0.1);
  gui.add(parameters, 'reset');


  let requestId = null;
  function reset() {
    if (requestId !== null) {
      cancelAnimationFrame(requestId);
    }

    const boidNum = parameters.static['boid num'];
    let boidTextureSize;
    let boidNumN;
    for (let i = 0; ; i++) {
      boidTextureSize = 2 ** i;
      if (boidTextureSize * boidTextureSize > boidNum) {
        boidNumN = i * 2;
        break;
      }
    }

    const separationRadius = parameters.static['separation radius'];
    const alignmentRadius = parameters.static['alignment radius'];
    const cohesionRadius = parameters.static['cohesion radius'];
    const neighborRadius = Math.max(separationRadius, alignmentRadius, cohesionRadius);
    const bucketSize = Math.ceil(1.0 / (2.0 * neighborRadius));
    let bucketReferrerTextureSize;
    for (let i = 0; ; i++) {
      bucketReferrerTextureSize = 2 ** i;
      if (bucketReferrerTextureSize > bucketSize) {
        break;
      }
    }

    let boidFbObjR = createBoidFramebuffer(gl, boidTextureSize);
    let boidFbObjW = createBoidFramebuffer(gl, boidTextureSize);
    const swapBoidFbObj = function() {
      const tmp = boidFbObjR;
      boidFbObjR = boidFbObjW;
      boidFbObjW = tmp;
    };
    let forceFbObj = createForceFramebuffer(gl, boidTextureSize);
    let bucketFbObjR = createBucketFramebuffer(gl, boidTextureSize);
    let bucketFbObjW = createBucketFramebuffer(gl, boidTextureSize);
    const swapBucketFbObj = function() {
      const tmp = bucketFbObjR;
      bucketFbObjR = bucketFbObjW;
      bucketFbObjW = tmp;
    }
    const bucketReferrerFbObj = createBucketReferrerFramebuffer(gl, bucketReferrerTextureSize);

    const initializeBoids = function() {
      gl.bindFramebuffer(gl.FRAMEBUFFER, boidFbObjW.framebuffer);
      gl.viewport(0.0, 0.0, boidTextureSize, boidTextureSize);
  
      gl.useProgram(initializeBoidProgram);
      gl.uniform2f(initializeBoidUniforms['u_randomSeed'], Math.random() * 1000.0, Math.random() * 1000.0);
      gl.uniform1ui(initializeBoidUniforms['u_boidNum'], boidNum);
      gl.uniform1ui(initializeBoidUniforms['u_boidTextureSize'], boidTextureSize)
      gl.uniform1f(initializeBoidUniforms['u_maxSpeed'], parameters.dynamic['max speed']);

      gl.bindVertexArray(fillScreenVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
  
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapBoidFbObj();
    };
  
    const initializeBucket = function() {
      gl.bindFramebuffer(gl.FRAMEBUFFER, bucketFbObjW.framebuffer);
      gl.viewport(0.0, 0.0, boidTextureSize, boidTextureSize);
      gl.useProgram(initializeBucketProgram);
      setTextureAsUniform(gl, 0, boidFbObjR.positionTexture, initializeBucketUniforms['u_positionTexture']);
      gl.uniform1f(initializeBucketUniforms['u_neighborRadius'], neighborRadius);
      gl.uniform1ui(initializeBucketUniforms['u_boidNum'], boidNum);
      gl.bindVertexArray(fillScreenVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapBucketFbObj();
    }
  
    const swapBucketIndex = function(i, j) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, bucketFbObjW.framebuffer);
      gl.viewport(0.0, 0.0, boidTextureSize, boidTextureSize);
      gl.useProgram(swapBucketIndexProgram);
      setTextureAsUniform(gl, 0, bucketFbObjR.bucketTexture, swapBucketIndexUniforms['u_bucketTexture']);
      gl.uniform1ui(swapBucketIndexUniforms['u_size'], boidTextureSize, boidTextureSize);
      gl.uniform1ui(swapBucketIndexUniforms['u_blockStep'], i);
      gl.uniform1ui(swapBucketIndexUniforms['u_subBlockStep'], j);
      gl.bindVertexArray(fillScreenVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
  
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapBucketFbObj();
    }
  
    const initializeBucketRefrrer = function() {
      gl.bindFramebuffer(gl.FRAMEBUFFER, bucketReferrerFbObj.framebuffer);
      gl.viewport(0.0, 0.0, bucketReferrerTextureSize, bucketReferrerTextureSize);
      gl.useProgram(initializeBucketReferrerProgram);
      setTextureAsUniform(gl, 0, bucketFbObjR.bucketTexture, initializeBucketReferrerUniforms['u_bucketTexture']);
      gl.uniform1f(initializeBucketReferrerUniforms['u_neighborRadius'], neighborRadius);
      gl.uniform1i(initializeBucketReferrerUniforms['u_boidNumN'], boidNumN);
      gl.uniform2i(initializeBucketReferrerUniforms['u_bucketReferrerTextureSize'], bucketReferrerTextureSize, bucketReferrerTextureSize);
      gl.bindVertexArray(fillScreenVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
  
    const constructBuckets = function() {
      initializeBucket();
      // sort by bitonic sort
      for (let i = 0; i < boidNumN; i++) {
        for (let j = 0; j <= i; j++) {
          swapBucketIndex(i, j);
        }
      }
      initializeBucketRefrrer();
    }

    const computeForces = function() {
      gl.bindFramebuffer(gl.FRAMEBUFFER, forceFbObj.framebuffer);
      gl.viewport(0.0, 0.0, boidTextureSize, boidTextureSize);
      gl.useProgram(computeForceProgram);
      setTextureAsUniform(gl, 0, boidFbObjR.positionTexture, computeForceUniforms['u_positionTexture']);
      setTextureAsUniform(gl, 1, boidFbObjR.velocityTexture, computeForceUniforms['u_velocityTexture']);
      setTextureAsUniform(gl, 2, bucketFbObjR.bucketTexture, computeForceUniforms['u_bucketTexture']);
      setTextureAsUniform(gl, 3, bucketReferrerFbObj.bucketReferrerTexture, computeForceUniforms['u_bucketReferrerTexture']);
      gl.uniform1ui(computeForceUniforms['u_boidNum'], boidNum);
      gl.uniform1ui(computeForceUniforms['u_boidTextureSize'], boidTextureSize);
      gl.uniform1f(computeForceUniforms['u_maxSpeed'], parameters.dynamic['max speed']);
      gl.uniform1f(computeForceUniforms['u_maxForce'], parameters.dynamic['max force']);
      gl.uniform1f(computeForceUniforms['u_bucketSize'], 2.0 * neighborRadius);
      gl.uniform3f(computeForceUniforms['u_radius'], separationRadius, alignmentRadius, cohesionRadius);
      gl.uniform3f(computeForceUniforms['u_weight'],
        parameters.dynamic['separation weight'], parameters.dynamic['alignment weight'], parameters.dynamic['cohesion weight']);
      gl.bindVertexArray(fillScreenVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    const updateBoids = function(deltaTime) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, boidFbObjW.framebuffer);
      gl.viewport(0.0, 0.0, boidTextureSize, boidTextureSize);
      gl.useProgram(updateBoidProgram);
      setTextureAsUniform(gl, 0, boidFbObjR.positionTexture, updateBoidUniforms['u_positionTexture']);
      setTextureAsUniform(gl, 1, boidFbObjR.velocityTexture, updateBoidUniforms['u_velocityTexture']);
      setTextureAsUniform(gl, 2, forceFbObj.forceTexture, updateBoidUniforms['u_forceTexture']);
      gl.uniform1f(updateBoidUniforms['u_deltaTime'], deltaTime);
      gl.uniform1ui(updateBoidUniforms['u_boidNum'], boidNum);
      gl.uniform1ui(updateBoidUniforms['u_boidTextureSize'], boidTextureSize);
      gl.uniform1f(updateBoidUniforms['u_maxSpeed'], parameters.dynamic['max speed']);
      gl.bindVertexArray(fillScreenVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapBoidFbObj();
    }

    const stepSimulation = function(deltaTime) {
      constructBuckets();
      computeForces();
      updateBoids(deltaTime);
    };

    const renderBoids = function() {
      gl.viewport(0.0, 0.0, canvas.width, canvas.height);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(renderBoidProgram);
      setTextureAsUniform(gl, 0, boidFbObjR.positionTexture, renderBoidUniforms['u_positionTexture']);
      setTextureAsUniform(gl, 1, boidFbObjR.velocityTexture, renderBoidUniforms['u_velocityTexture']);
      gl.uniform2f(renderBoidUniforms['u_canvasSize'], canvas.width, canvas.height);
      gl.uniform1f(renderBoidUniforms['u_boidSize'], parameters.dynamic['boid size']);

      gl.drawArrays(gl.POINTS, 0, boidNum);
    }

    initializeBoids();
  
    gl.clearColor(0.2, 0.2, 0.2, 1.0);
    let previousTime = performance.now();
    const render = function() {
      stats.update();
  
      const currentTime = performance.now();
      const deltaTime = Math.min(0.05, (currentTime - previousTime) * 0.001);
      previousTime = currentTime;


      stepSimulation(deltaTime);
      renderBoids();

      requestId = requestAnimationFrame(render);
    };
    render();
  };
  reset();


}());