(function() {

  function createBoidMesh() {
    const p0 = new Vector3(0.0, 0.3, -1.2);
    const p1 = new Vector3(-0.5, 0.3, 0.8);
    const p2 = new Vector3(0.5, 0.3, 0.5);
    const p3 = new Vector3(0.0, -0.3, 0.0);

    const n0 = Vector3.cross(Vector3.sub(p1, p0), Vector3.sub(p2, p0)).norm();
    const n1 = Vector3.cross(Vector3.sub(p3, p0), Vector3.sub(p1, p0)).norm();
    const n2 = Vector3.cross(Vector3.sub(p2, p0), Vector3.sub(p3, p0)).norm();
    const n3 = Vector3.cross(Vector3.sub(p3, p1), Vector3.sub(p2, p1)).norm();

    const positions = new Float32Array([
      p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z,
      p0.x, p0.y, p0.z, p3.x, p3.y, p3.z, p1.x, p1.y, p1.z,
      p0.x, p0.y, p0.z, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z,
      p1.x, p1.y, p1.z, p3.x, p3.y, p3.z, p2.x, p2.y, p2.z
    ]);

    const normals = new Float32Array([
      n0.x, n0.y, n0.z, n0.x, n0.y, n0.z, n0.x, n0.y, n0.z,
      n1.x, n1.y, n1.z, n1.x, n1.y, n1.z, n1.x, n1.y, n1.z,
      n2.x, n2.y, n2.z, n2.x, n2.y, n2.z, n2.x, n2.y, n2.z,
      n3.x, n3.y, n3.z, n3.x, n3.y, n3.z, n3.x, n3.y, n3.z
    ]);

    return {
      positions: positions,
      normals: normals
    };
  }

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

layout (location = 0) out vec4 o_position;
layout (location = 1) out vec4 o_velocity;

uniform vec2 u_randomSeed;
uniform uint u_boidNum;
uniform uint u_boidTextureSize;
uniform float u_maxSpeed;
uniform vec3 u_simulationSpace;

uint convertCoordToIndex(uvec2 coord, uint sizeX) {
  return coord.x + sizeX * coord.y;
}

float random(vec2 x){
  return fract(sin(dot(x,vec2(12.9898, 78.233))) * 43758.5453);
}

void main(void) {
  uint index = convertCoordToIndex(uvec2(gl_FragCoord.xy), u_boidTextureSize);

  if (index >= u_boidNum) { // unused pixels
    o_position = vec4(0.0);
    o_velocity = vec4(0.0);
    return;
  }

  vec3 position = vec3(
    random(gl_FragCoord.xy * 0.013 + random(u_randomSeed * vec2(32.19, 27.51) * 1000.0)),
    random(gl_FragCoord.xy * 0.029 + random(u_randomSeed * vec2(19.56, 11.34) * 1000.0)),
    random(gl_FragCoord.xy * 0.018 + random(u_randomSeed * vec2(41.71, 25.93) * 1000.0))
  ) * (u_simulationSpace - 1e-5) + 1e-5 * 0.5;
  vec3 velocity = normalize(vec3(
    random(gl_FragCoord.xy * 0.059 + random(u_randomSeed * vec2(27.31, 16.91) * 1000.0)),
    random(gl_FragCoord.xy * 0.038 + random(u_randomSeed * vec2(25.95, 19.47) * 1000.0)),
    random(gl_FragCoord.xy * 0.029 + random(u_randomSeed * vec2(36.19, 27.33) * 1000.0))
  ) * 2.0 - 1.0) * u_maxSpeed;

  o_position = vec4(position, 0.0);
  o_velocity = vec4(velocity, 0.0);
}
`

  const UPDATE_BOID_FRAGMENT_SHADER_SOURCE = 
`#version 300 es

precision highp float;

layout (location = 0) out vec4 o_position;
layout (location = 1) out vec4 o_velocity;

uniform sampler2D u_positionTexture;
uniform sampler2D u_velocityTexture;
uniform sampler2D u_forceTexture;
uniform float u_deltaTime;
uniform float u_maxSpeed;
uniform uint u_boidNum;
uniform uint u_boidTextureSize;
uniform vec3 u_simulationSpace;

uint convertCoordToIndex(uvec2 coord, uint sizeX) {
  return coord.x + sizeX * coord.y;
}

vec3 limit(vec3 v, float max) {
  if (length(v) < max) {
    return normalize(v) * max;
  }
  return v;
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);

  uint index = convertCoordToIndex(uvec2(coord), u_boidTextureSize);
  if (index >= u_boidNum) { // unused pixels
    o_position = vec4(0.0);
    o_velocity = vec4(0.0);
    return;
  }

  vec3 position = texelFetch(u_positionTexture, coord, 0).xyz;
  vec3 velocity = texelFetch(u_velocityTexture, coord, 0).xyz;
  vec3 force = texelFetch(u_forceTexture, coord, 0).xyz;

  vec3 nextVelocity = limit(velocity + u_deltaTime * force, u_maxSpeed);
  vec3 nextPosition = position + u_deltaTime * velocity;

  if (nextPosition.x < 1e-6) {
    nextPosition.x = 1e-6;
    nextVelocity.x *= -1.0;
  }
  if (nextPosition.x > u_simulationSpace.x - 1e-6) {
    nextPosition.x = u_simulationSpace.x - 1e-6;
    nextVelocity.x *= -1.0;
  }
  if (nextPosition.y < 1e-6) {
    nextPosition.y = 1e-6;
    nextVelocity.y *= -1.0;
  }
  if (nextPosition.y > u_simulationSpace.y - 1e-6) {
    nextPosition.y = u_simulationSpace.y - 1e-6;
    nextVelocity.y *= -1.0;
  }
  if (nextPosition.z < 1e-6) {
    nextPosition.z = 1e-6;
    nextVelocity.z *= -1.0;
  }
  if (nextPosition.z > u_simulationSpace.z - 1e-6) {
    nextPosition.z = u_simulationSpace.z - 1e-6;
    nextVelocity.z *= -1.0;
  }

  o_position = vec4(nextPosition, 0.0);
  o_velocity = vec4(nextVelocity, 0.0);
}
`

const COMPUTE_FORCE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;
precision highp isampler2D;
precision highp usampler2D;

// 4294967295 = 2^32 - 1
#define MAX_32UI 4294967295u

out vec4 o_force;

uniform sampler2D u_positionTexture;
uniform sampler2D u_velocityTexture;
uniform usampler2D u_bucketTexture;
uniform usampler2D u_bucketReferrerTexture;

uniform vec3 u_radius; // x: separation, y: alignment: z: cohesion
uniform vec3 u_weight; // x: separation, y: alignment: z: cohesion
uniform float u_maxSpeed;
uniform float u_maxForce;
uniform float u_boundaryIntensity;
uniform uint u_boidNum;
uniform float u_bucketSize;
uniform ivec3 u_bucketNum;
uniform vec3 u_simulationSpace;

uint convertCoordToIndex(uvec2 coord, uint sizeX) {
  return coord.x + sizeX * coord.y;
}

uvec2 convertIndexToCoord(uint index, uint sizeX) {
  return uvec2(index % sizeX, index / sizeX);
}

vec3 limit(vec3 v, float max) {
  if (length(v) > max) {
    return normalize(v) * max;
  }
  return v;
}

void findNeighbors(vec3 position, ivec3 bucketPosition, ivec3 bucketNum, uint boidTextureSizeX, uint bucketReferrerTextureSizeX, inout vec3 separation, inout vec3 alignment, inout vec4 cohesion) {
  if (bucketPosition.x < 0 || bucketPosition.x >= bucketNum.x ||
      bucketPosition.y < 0 || bucketPosition.y >= bucketNum.y ||
      bucketPosition.z < 0 || bucketPosition.z >= bucketNum.z) {
    return;
  }
  uint bucketIndex = uint(bucketPosition.x + bucketNum.x * bucketPosition.y + (bucketNum.x * bucketNum.y) * bucketPosition.z);
  ivec2 coord = ivec2(convertIndexToCoord(bucketIndex, bucketReferrerTextureSizeX));

  uvec2 bucketReferrer = texelFetch(u_bucketReferrerTexture, coord, 0).xy;

  if (bucketReferrer.x == MAX_32UI || bucketReferrer.y == MAX_32UI) {
    return;
  }

  for (uint i = bucketReferrer.x; i <= bucketReferrer.y; i++) {
    uint boidIndex = texelFetch(u_bucketTexture, ivec2(convertIndexToCoord(i, boidTextureSizeX)), 0).y;

    ivec2 boidCoord = ivec2(convertIndexToCoord(boidIndex, boidTextureSizeX));
    vec3 otherPos = texelFetch(u_positionTexture, boidCoord, 0).xyz;

    float dist = length(position - otherPos);

    if (dist == 0.0) {
      continue;
    }

    if (dist < u_radius.x) {
      separation += normalize(position - otherPos) / dist;
    }

    if (dist < u_radius.y) {
      vec3 otherVel = texelFetch(u_velocityTexture, boidCoord, 0).xyz;
      alignment += otherVel;
    }

    if (dist < u_radius.z) {
      cohesion.xyz += otherPos;
      cohesion.w += 1.0;
    }
  }
}

vec3 computeForceFromBoids(vec3 position, vec3 velocity, uint boidTextureSizeX) {
  uint bucketReferrerTextureSizeX = uint(textureSize(u_bucketReferrerTexture, 0).x);

  vec3 bucketPosition = position / u_bucketSize;
  int xOffset = fract(bucketPosition.x) < 0.5 ? -1 : 1;
  int yOffset = fract(bucketPosition.y) < 0.5 ? -1 : 1;
  int zOffset = fract(bucketPosition.z) < 0.5 ? -1 : 1;

  ivec3 bucketPosition000 = ivec3(bucketPosition);
  ivec3 bucketPosition100 = bucketPosition000 + ivec3(xOffset, 0, 0);
  ivec3 bucketPosition010 = bucketPosition000 + ivec3(0, yOffset, 0);
  ivec3 bucketPosition110 = bucketPosition000 + ivec3(xOffset, yOffset, 0);
  ivec3 bucketPosition001 = bucketPosition000 + ivec3(0, 0, zOffset);
  ivec3 bucketPosition101 = bucketPosition000 + ivec3(xOffset, 0, zOffset);
  ivec3 bucketPosition011 = bucketPosition000 + ivec3(0, yOffset, zOffset);
  ivec3 bucketPosition111 = bucketPosition000 + ivec3(xOffset, yOffset, zOffset);

  vec3 separation = vec3(0.0);
  vec3 alignment = vec3(0.0);
  vec4 cohesion = vec4(0.0);
  findNeighbors(position, bucketPosition000, u_bucketNum, boidTextureSizeX, bucketReferrerTextureSizeX, separation, alignment, cohesion);
  findNeighbors(position, bucketPosition100, u_bucketNum, boidTextureSizeX, bucketReferrerTextureSizeX, separation, alignment, cohesion);
  findNeighbors(position, bucketPosition010, u_bucketNum, boidTextureSizeX, bucketReferrerTextureSizeX, separation, alignment, cohesion);
  findNeighbors(position, bucketPosition110, u_bucketNum, boidTextureSizeX, bucketReferrerTextureSizeX, separation, alignment, cohesion);
  findNeighbors(position, bucketPosition001, u_bucketNum, boidTextureSizeX, bucketReferrerTextureSizeX, separation, alignment, cohesion);
  findNeighbors(position, bucketPosition101, u_bucketNum, boidTextureSizeX, bucketReferrerTextureSizeX, separation, alignment, cohesion);
  findNeighbors(position, bucketPosition011, u_bucketNum, boidTextureSizeX, bucketReferrerTextureSizeX, separation, alignment, cohesion);
  findNeighbors(position, bucketPosition111, u_bucketNum, boidTextureSizeX, bucketReferrerTextureSizeX, separation, alignment, cohesion);


  vec3 separationForce = vec3(0.0);
  if (separation != vec3(0.0)) {
    vec3 desiredVelocity = normalize(separation) * u_maxSpeed;
    separationForce = limit(desiredVelocity - velocity, u_maxForce);
  }

  vec3 alignmentForce = vec3(0.0);
  if (alignment != vec3(0.0)) {
    vec3 desiredVelocity = normalize(alignment) * u_maxSpeed;
    alignmentForce = limit(desiredVelocity - velocity, u_maxForce);
  }

  vec3 cohesionForce = vec3(0.0);
  if (cohesion.w != 0.0) {
    vec3 target = cohesion.xyz / cohesion.w;
    vec3 desiredVelocity = normalize(target - position) * u_maxSpeed;
    cohesionForce = limit(desiredVelocity - velocity, u_maxForce);
  }

  separationForce *= u_weight.x;
  alignmentForce *= u_weight.y;
  cohesionForce *= u_weight.z;

  return separationForce + alignmentForce + cohesionForce;
}

vec3 computeForceFromWalls(vec3 position) {
  vec3 force = vec3(0.0);
  if (position.x < u_radius.x) {
    force += vec3(u_boundaryIntensity, 0.0, 0.0) / position.x;
  }
  if (u_simulationSpace.x - position.x < u_radius.x) {
    force += vec3(-u_boundaryIntensity, 0.0, 0.0) / (u_simulationSpace.x - position.x);
  }
  if (position.y < u_radius.x) {
    force += vec3(0.0, u_boundaryIntensity, 0.0) / position.y;
  }
  if (u_simulationSpace.y - position.y < u_radius.x) {
    force += vec3(0.0, -u_boundaryIntensity, 0.0) / (u_simulationSpace.y - position.y);
  }
  if (position.z < u_radius.x) {
    force += vec3(0.0, 0.0, u_boundaryIntensity) / position.z;
  }
  if (u_simulationSpace.z - position.z < u_radius.x) {
    force += vec3(0.0, 0.0, -u_boundaryIntensity) / (u_simulationSpace.z- position.z);
  }
  return force;
}

vec3 computeForce(uint boidTextureSizeX) {

  ivec2 coord = ivec2(gl_FragCoord.xy);
  vec3 position = texelFetch(u_positionTexture, coord, 0).xyz;
  vec3 velocity = texelFetch(u_velocityTexture, coord, 0).xyz;

  vec3 forceFromBoids = computeForceFromBoids(position, velocity, boidTextureSizeX);
  vec3 forceFromWalls = computeForceFromWalls(position);

  return limit(forceFromBoids + forceFromWalls, u_maxForce);
}

void main(void) {
  uint boidTextureSizeX = uint(textureSize(u_positionTexture, 0).x);
  uint index = convertCoordToIndex(uvec2(gl_FragCoord.xy), boidTextureSizeX);
  if (index >= u_boidNum) { // unused pixels
    o_force = vec4(0.0);
    return;
  }
  vec3 force = computeForce(boidTextureSizeX);
  o_force = vec4(force, 0.0);
}
`;

  const RENDER_BOID_VERTEX_SHADER_SOURCE =
`#version 300 es

precision highp isampler2D;
precision highp usampler2D;

layout (location = 0) in vec3 i_position;
layout (location = 1) in vec3 i_normal;

out vec3 v_normal;

uniform sampler2D u_positionTexture;
uniform sampler2D u_velocityTexture;
uniform mat4 u_mvpMatrix;
uniform vec3 u_simulationSpace;
uniform float u_boidSize;

ivec2 convertIndexToCoord(int index, int sizeX) {
  return ivec2(index % sizeX, index / sizeX);
}

mat3 getLookMat(ivec2 coord) {
  vec3 velocity = texelFetch(u_velocityTexture, coord, 0).xyz;

  vec3 z = -normalize(velocity);
  vec3 x = cross(vec3(0.0, 1.0, 0.0), z);
  vec3 y = cross(z, x);

  return mat3(
    x.x, x.y, x.z,
    y.x, y.y, y.z,
    z.x, z.y, z.z
  );
}

void main(void) {
  ivec2 coord = convertIndexToCoord(gl_InstanceID, textureSize(u_positionTexture, 0).x);
  mat3 lookMat = getLookMat(coord);
  vec3 instancePosition = texelFetch(u_positionTexture, coord, 0).xyz;
  vec3 position = lookMat * (i_position * u_boidSize) + (2.0 * instancePosition - u_simulationSpace) * 300.0;

  gl_Position = u_mvpMatrix * vec4(position, 1.0);
  v_normal = (u_mvpMatrix * vec4(i_normal, 0.0)).xyz;
}
`

  const RENDER_BOID_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec3 v_normal;

out vec4 o_color;

vec3 LightDir1 = normalize(vec3(1.0, 1.0, 1.0));
vec3 LightDir2 = normalize(vec3(-0.8, -0.5, -1.0));
vec3 LightDir3 = normalize(vec3(-0.3, 0.2, -0.7));

void main(void) {
  vec3 normal = normalize(v_normal);
  vec3 diffuse1 = vec3(0.3) * max(0.0, dot(normal, LightDir1));
  vec3 diffuse2 = vec3(0.2) * max(0.0, dot(normal, LightDir2));
  vec3 diffuse3 = vec3(0.15) * max(0.0, dot(normal, LightDir3));
  o_color = vec4(diffuse1 + diffuse2 + diffuse3, 1.0);
}
`

const INITIALIZE_BUCKET_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

// 4294967295 = 2^32 - 1
#define MAX_32UI 4294967295u

out uvec2 o_bucket;

uniform sampler2D u_positionTexture;
uniform float u_bucketSize;
uniform uint u_boidNum;
uniform uvec3 u_bucketNum;

uint convertCoordToIndex(uvec2 coord, uint sizeX) {
  return coord.x + sizeX * coord.y;
}

uint getBucketIndex(vec3 position) {
  uvec3 bucketCoord = uvec3(position / u_bucketSize);
  return bucketCoord.x + bucketCoord.y * u_bucketNum.x + bucketCoord.z * (u_bucketNum.x * u_bucketNum.y);
}

void main(void) {
  uint positionTextureSizeX = uint(textureSize(u_positionTexture, 0).x);
  uint boidIndex = convertCoordToIndex(uvec2(gl_FragCoord.xy), positionTextureSizeX);
  if (boidIndex >= u_boidNum) {
    o_bucket = uvec2(MAX_32UI, 0); // = uvec2(2^32 - 1, 0)
  }
  vec3 position = texelFetch(u_positionTexture, ivec2(gl_FragCoord.xy), 0).xyz;
  uint bucketIndex = getBucketIndex(position);
  o_bucket = uvec2(bucketIndex, boidIndex);
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

// 4294967295 = 2^32 - 1
#define MAX_32UI 4294967295u

out uvec2 o_referrer;

uniform uvec2 u_bucketReferrerTextureSize;
uniform usampler2D u_bucketTexture;
uniform uint u_boidNumN;
uniform uvec3 u_bucketNum;

uint convertCoordToIndex(uvec2 coord, uint sizeX) {
  return coord.x + sizeX * coord.y;
}

uvec2 convertIndexToCoord(uint index, uint sizeX) {
  return uvec2(index % sizeX, index / sizeX);
}

uint getBucketIndex(uint particleIndex, uint particleTextureSizeX) {
  return texelFetch(u_bucketTexture, ivec2(convertIndexToCoord(particleIndex, particleTextureSizeX)), 0).x;
}

uint binarySearchMinIndex(uint target, uint from, uint to, uint particleTextureSizeX) {
  for (uint i = 0u; i < u_boidNumN + 1u; i++) {
    uint middle = from + (to - from) / 2u;
    uint bucketIndex = getBucketIndex(middle, particleTextureSizeX);
    if (bucketIndex < target) {
      from = middle + 1u;
    } else {
      to = middle;
    }
    if (from == to) {
      if (getBucketIndex(from, particleTextureSizeX) == target) {
        return from;
      } else {
        return MAX_32UI;
      }
    }
  }
  return MAX_32UI;
}

uint binarySearchMaxIndex(uint target, uint from, uint to, uint particleTextureSizeX) {
  for (uint i = 0u; i < u_boidNumN + 1u; i++) {
    uint middle = from + (to - from) / 2u + 1u;
    uint bucketIndex = getBucketIndex(middle, particleTextureSizeX);
    if (bucketIndex > target) {
      to = middle - 1u;
    } else {
      from = middle;
    }
    if (from == to) {
      if (getBucketIndex(from, particleTextureSizeX) == target) {
        return from;
      } else {
        return MAX_32UI;
      }
    }
  }
  return MAX_32UI;
}

uvec2 binarySearchRange(uint target, uint from, uint to) {
  uint boidTextureSizeX = uint(textureSize(u_bucketTexture, 0).x);
  from =  binarySearchMinIndex(target, from, to, boidTextureSizeX);
  to = from == MAX_32UI ? MAX_32UI : binarySearchMaxIndex(target, from, to, boidTextureSizeX);
  return uvec2(from, to);
}

void main(void) {
  uint bucketIndex = convertCoordToIndex(uvec2(gl_FragCoord.xy), u_bucketReferrerTextureSize.x);
  uint maxBucketIndex = u_bucketNum.x * u_bucketNum.y * u_bucketNum.z;

  if (bucketIndex >= maxBucketIndex) {
    o_referrer = uvec2(MAX_32UI, MAX_32UI);
    return;
  }

  uvec2 boidTextureSize = uvec2(textureSize(u_bucketTexture, 0));
  uint boidNum = boidTextureSize.x * boidTextureSize.y;

  o_referrer = binarySearchRange(bucketIndex, 0u, boidNum - 1u);
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
    const positionTexture = createTexture(gl, size, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, positionTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, positionTexture, 0);
    const velocityTexture = createTexture(gl, size, gl.RGBA32F, gl.RGBA, gl.FLOAT);
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
    const forceTexture = createTexture(gl, size, gl.RGBA32F, gl.RGBA, gl.FLOAT);
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
    const bucketReferrerTexture = createTexture(gl, size, gl.RG32UI, gl.RG_INTEGER, gl.UNSIGNED_INT);
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
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.CULL_FACE);

  const initializeBoidProgram = createInitializeBoidProgram(gl);
  const updateBoidProgram = createUpdateBoidProgram(gl);
  const computeForceProgram = createComputeForceProgram(gl);
  const renderBoidProgram = createRenderBoidProgram(gl);
  const initializeBucketProgram = createInitializeBucketProgram(gl);
  const swapBucketIndexProgram = createSwapBucketIndexProgram(gl);
  const initializeBucketReferrerProgram = createInitializeBucketReferrerProgram(gl);

  const initializeBoidUniforms = getUniformLocations(gl, initializeBoidProgram, ['u_randomSeed', 'u_boidNum', 'u_boidTextureSize', 'u_maxSpeed', 'u_simulationSpace']);
  const updateBoidUniforms = getUniformLocations(gl, updateBoidProgram, ['u_positionTexture', 'u_velocityTexture', 'u_forceTexture', 'u_deltaTime', 'u_boidNum', 'u_boidTextureSize', 'u_maxSpeed', 'u_simulationSpace']);
  const computeForceUniforms = getUniformLocations(gl, computeForceProgram,
    ['u_positionTexture', 'u_velocityTexture', 'u_bucketTexture', 'u_bucketReferrerTexture', 'u_boidNum', 'u_radius', 'u_weight', 'u_maxSpeed', 'u_maxForce', 'u_boundaryIntensity', 'u_bucketSize', 'u_bucketNum', 'u_simulationSpace']);
  const renderBoidUniforms = getUniformLocations(gl, renderBoidProgram, ['u_positionTexture', 'u_velocityTexture', 'u_mvpMatrix', 'u_simulationSpace', 'u_boidSize']);
  const initializeBucketUniforms = getUniformLocations(gl, initializeBucketProgram, ['u_positionTexture', 'u_bucketSize', 'u_boidNum', 'u_bucketNum']);
  const swapBucketIndexUniforms = getUniformLocations(gl, swapBucketIndexProgram, ['u_bucketTexture', 'u_size', 'u_blockStep', 'u_subBlockStep']);
  const initializeBucketReferrerUniforms = getUniformLocations(gl, initializeBucketReferrerProgram, ['u_bucketTexture', 'u_bucketReferrerTextureSize', 'u_boidNumN', 'u_bucketNum']);

  const fillScreenVao = createVao(gl,
    [{buffer: createVbo(gl, VERTICES_POSITION), size: 2, index: 0}],
    createIbo(gl, VERTICES_INDEX)
  );

  const stats = new Stats();
  document.body.appendChild(stats.dom);

  const MAX_BOID_NUM = 4294967295; // = 2^32 - 1
　const MAX_BUCKET_NUM = 4294967295; // = 2^32 - 1

  const gui = new dat.GUI();
  const parameters = {
    dynamic: {
      'separation weight': 1.5,
      'alignment weight': 1.0,
      'cohesion weight': 1.0,
      'max speed': 0.05,
      'max force': 0.1,
      'boundary intensity': 30.0,
      'boid size': 10.0,
    },
    static: {
      'boid num': 4096,
      'separation radius': 0.05,
      'alignment radius': 0.075,
      'cohesion radius': 0.10,
    },
    camera: {
      'angle': -25.0,
      'distance': 1000.0,
      'height': 250.0
    },
    'reset': () => reset()
  };
  const dynamicFolder = gui.addFolder('dynamic parameters');
  dynamicFolder.add(parameters.dynamic, 'separation weight', 0.0, 5.0);
  dynamicFolder.add(parameters.dynamic, 'alignment weight', 0.0, 5.0);
  dynamicFolder.add(parameters.dynamic, 'cohesion weight', 0.0, 5.0);
  dynamicFolder.add(parameters.dynamic, 'max speed', 0.0, 0.2);
  dynamicFolder.add(parameters.dynamic, 'max force', 0.0, 1.0);
  dynamicFolder.add(parameters.dynamic, 'boundary intensity', 0.0, 50.0);
  dynamicFolder.add(parameters.dynamic, 'boid size', 1.0, 50.0);
  const staticFolder = gui.addFolder('static parameters');
  staticFolder.add(parameters.static, 'boid num', 1, 16384);
  staticFolder.add(parameters.static, 'separation radius', 0.01, 0.2);
  staticFolder.add(parameters.static, 'alignment radius', 0.01, 0.2);
  staticFolder.add(parameters.static, 'cohesion radius', 0.01, 0.2);
  const cameraFolder = gui.addFolder('camera');
  cameraFolder.add(parameters.camera, 'angle', -180, 180);
  cameraFolder.add(parameters.camera, 'distance', 50.0, 3000.0);
  cameraFolder.add(parameters.camera, 'height', -3000.0, 3000.0);
  gui.add(parameters, 'reset');

  let requestId = null;
  function reset() {
    if (requestId !== null) {
      cancelAnimationFrame(requestId);
    }

    const boidMesh = createBoidMesh();
    const boidPositionVbo = createVbo(gl, boidMesh.positions);
    const boidNormalVbo = createVbo(gl, boidMesh.normals);

    const boidNum = parameters.static['boid num'];
    if (boidNum > MAX_BOID_NUM) {
      throw new Error(`number of boids must be less than ${MAX_BOID_NUM}. current value is ${boidNum}.`);
    }
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
    const bucketSize = 2.0 * neighborRadius;
    const simulationSpace = new Vector3(1.6, 1.2, 1.6);
    const bucketNum = Vector3.div(simulationSpace, bucketSize).ceil().add(new Vector3(1, 1, 1));
    const totalBuckets = bucketNum.x * bucketNum.y * bucketNum.z;
    if (totalBuckets > MAX_BUCKET_NUM) {
      throw new Error(`number of buckets must be less than ${MAX_BUCKET_NUM}. current value is ${totalBuckets}.`);
    }
    let bucketReferrerTextureSize;
    for (let i = 0; ; i++) {
      bucketReferrerTextureSize = 2 ** i;
      if (bucketReferrerTextureSize * bucketReferrerTextureSize > totalBuckets) {
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
      gl.uniform3f(initializeBoidUniforms['u_simulationSpace'], simulationSpace.x, simulationSpace.y, simulationSpace.z);
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
      gl.uniform1f(initializeBucketUniforms['u_bucketSize'], bucketSize);
      gl.uniform1ui(initializeBucketUniforms['u_boidNum'], boidNum);
      gl.uniform3ui(initializeBucketUniforms['u_bucketNum'], bucketNum.x, bucketNum.y, bucketNum.z);
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
      gl.uniform1ui(initializeBucketReferrerUniforms['u_boidNumN'], boidNumN);
      gl.uniform2ui(initializeBucketReferrerUniforms['u_bucketReferrerTextureSize'], bucketReferrerTextureSize, bucketReferrerTextureSize);
      gl.uniform3ui(initializeBucketReferrerUniforms['u_bucketNum'], bucketNum.x, bucketNum.y, bucketNum.z);
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
      gl.uniform1f(computeForceUniforms['u_maxSpeed'], parameters.dynamic['max speed']);
      gl.uniform1f(computeForceUniforms['u_maxForce'], parameters.dynamic['max force']);
      gl.uniform1f(computeForceUniforms['u_boundaryIntensity'], parameters.dynamic['boundary intensity']);
      gl.uniform1f(computeForceUniforms['u_bucketSize'], bucketSize);
      gl.uniform3i(computeForceUniforms['u_bucketNum'], bucketNum.x, bucketNum.y, bucketNum.z);
      gl.uniform3f(computeForceUniforms['u_simulationSpace'], simulationSpace.x, simulationSpace.y, simulationSpace.z);
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
      gl.uniform3f(updateBoidUniforms['u_simulationSpace'], simulationSpace.x, simulationSpace.y, simulationSpace.z);
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
      const cameraRadian = Math.PI * parameters.camera['angle'] / 180.0;
      const cameraPosition = new Vector3(
        parameters.camera['distance'] * Math.cos(cameraRadian),
        parameters.camera['height'],
        parameters.camera['distance'] * Math.sin(cameraRadian));
      const viewMatrix = Matrix4.inverse(Matrix4.lookAt(
        cameraPosition, Vector3.zero, new Vector3(0.0, 1.0, 0.0)
      ));
      const projectionMatrix = Matrix4.perspective(canvas.width / canvas.height, 60, 0.01, 10000.0);
      const mvpMatrix = Matrix4.mul(viewMatrix, projectionMatrix);

      gl.viewport(0.0, 0.0, canvas.width, canvas.height);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(renderBoidProgram);
      setTextureAsUniform(gl, 0, boidFbObjR.positionTexture, renderBoidUniforms['u_positionTexture']);
      setTextureAsUniform(gl, 1, boidFbObjR.velocityTexture, renderBoidUniforms['u_velocityTexture']);
      gl.uniformMatrix4fv(renderBoidUniforms['u_mvpMatrix'], false, mvpMatrix.elements);
      gl.uniform3f(renderBoidUniforms['u_simulationSpace'], simulationSpace.x, simulationSpace.y, simulationSpace.z);
      gl.uniform1f(renderBoidUniforms['u_boidSize'], parameters.dynamic['boid size']);

      [boidPositionVbo, boidNormalVbo].forEach((vbo, i) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.enableVertexAttribArray(i);
        gl.vertexAttribPointer(i, 3, gl.FLOAT, false, 0, 0);
      });
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 12, boidNum);
    }

    initializeBoids();
  
    gl.clearColor(1.0, 1.0, 0.98, 1.0);
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