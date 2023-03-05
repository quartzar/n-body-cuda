//
// Created by quartzar on 21/10/22.
//

#include <iostream>
#include <random>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include "NbodyRenderer.h"
#include "CONSTANTS.h"

#define GL_POINT_SPRITE_ARB               0x8861
#define GL_COORD_REPLACE_ARB              0x8862
#define GL_VERTEX_PROGRAM_POINT_SIZE_NV   0x8642

NbodyRenderer::NbodyRenderer()
: m_pos(nullptr),
m_vel(nullptr),
m_program(0),
m_texture(0),
m_vertexShader(0),
m_pixelShader(0),
m_vboColour(0),
m_pbo(0)
{
    // _initGL(); // unnecessary currently -> will need for shaders
}

NbodyRenderer::~NbodyRenderer()
{
    m_pos = nullptr;
}

void NbodyRenderer::setPositions(float *pos)
{
    m_pos = pos;
}

void NbodyRenderer::setVelocities(float *vel)
{
    m_vel = vel;
}

void NbodyRenderer::display(RenderMode mode, float zoom, float xRot,
                            float yRot, float zRot, float xTrans, float yTrans, float zTrans,
                            bool trailMode, bool colourMode)
{
    switch (mode)
    {
        case SPRITES: // TODO: implement sprites       // FROM NVIDIA
        {
            // std::cout << "\nSPRITES CALLED BUT NOT IMPLEMENTED YET";
            // for (int i = 0; i < N_BODIES; ++i) {
            //     const float4 pos = make_float4(m_hPos[i * 4], m_hPos[i * 4 + 1], m_hPos[i * 4 + 2], m_hPos[i * 4 + 3]);
            //     const float4 vel = make_float4(m_hVel[i * 4], m_hVel[i * 4 + 1], m_hVel[i * 4 + 2], m_hVel[i * 4 + 3]);
            //
            //     const float mass = pos.w; // mass is stored in the w component
            //
            //     drawSprite(pos.x, pos.y, mass * 0.01f, 1.0f, 1.0f, 1.0f, 1.0f);
            // }
            _drawSprite(1.f,1.f,1.f,1.f);
        }
            break;
        case POINTS:
        default:
        {
            if (trailMode)
                glClear(GL_DEPTH_BUFFER_BIT);
            else // CLEAR Z-BUFFER
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            // SET THE CAMERA ORIENTATION
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            gluLookAt(0, 0, 1, 0, 0, 0, 0, 1, 0);
    
            // ROTATE THE CAMERA B)
            // glTranslatef(0, 0, (-1.0f * (GLfloat)INIT_ZOOM));
            glTranslatef(xTrans, yTrans, (-1.0f * (GLfloat) INIT_ZOOM) + zTrans); // tx, ty, 0
            glRotatef(xRot, 0, 1, 0);
            glRotatef(yRot, 1, 0, 0);
            glRotatef(zRot, 0, 0, 1);
    
            // ZOOM THE CAMERA
            glScalef(zoom, zoom, zoom);
            // std::cout << "\nzoom => " << zoom << std::flush;
    
            glColor3f(1, 1, 1);
            glPointSize(ORB_SIZE);
            _drawOrbitals(colourMode);
        }
            break;
    }
}

void NbodyRenderer::_drawSprite(float r, float g, float b, float a)
{
    for (int i = 0; i < N_BODIES; i++) {
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glTranslatef(m_pos[4*i], m_pos[4*i+1], m_pos[4*i+2]);
    
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(r, g, b, a);
    
        const int sides = 16;
        const float step = 2.0f * M_PI / sides;
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(0.0f, 0.0f);
        float size = m_pos[4*i+3];
        for (int j = 0; j <= sides; j++) {
            const float angle = float(j) * step;
            const float px = size * std::cos(angle);
            const float py = size * std::sin(angle);
            glVertex2f(px, py);
        }
        glEnd();
    
        glDisable(GL_BLEND);
    
        glPopMatrix();
    }
}

void NbodyRenderer::_drawOrbitals(bool colour)
{
    if (!m_pbo)
    {
        glEnable( GL_POINT_SMOOTH);
        glEnable( GL_BLEND );   // the below setting is amazing for glowing dense regions
        glBlendFunc( GL_SRC_ALPHA, GL_DST_ALPHA);
        glBegin(GL_POINTS);
        {
            for (int i = 0; i < N_BODIES; i++) {
                // glPushMatrix(); // SAVES THE CURRENT VIEW MATRIX
                GLfloat *v[3] = {&m_pos[4 * i], &m_pos[4 * i + 1], &m_pos[4 * i + 2]};
                
                if (colour)
                {
                    // float normX = std::abs(m_pos[4*i]) / 10.f;
                    // float normY = std::abs(m_pos[4*i+1]) / 10.f;
                    float normX = m_vel[4*i] * 6.f;
                    float normY = m_vel[4*i+1] * 6.f;
                    float normZ = m_vel[4*i+2] * 6.f;
                    float r = sqrtf(normX*normX + normY*normY + normZ*normZ);
                    // float normZ = std::abs(m_pos[4*i+2]) / 10.0f;
                    // from https://en.wikipedia.org/wiki/SRGB#The_sRGB_transfer_function_.28.22gamma.22.29
                    float phi = r + normX * (2.f * (float)PI / 180.f);
                    float theta = r +  normY * (2.f * (float)PI / 180.f);
                    float psi = r +  normZ * (2.f * (float)PI / 180.f);
    
                    GLfloat sphPos[3];
                    sphPos[0] = cosf(phi) * cosf(theta);
                    sphPos[1] = sinf(phi) * cosf(theta);
                    sphPos[2] = sinf(psi);// + sinf(phi/2) + tanf(phi/4);
                    GLfloat col[3] =
                            {
                                    std::max(0.4f, 3.2404542f * sphPos[0] - 1.5371385f * sphPos[1] - 0.4985314f * sphPos[2]),
                                    std::max(0.3f, -0.9692660f * sphPos[0] + 1.8760108f * sphPos[1] + 0.0415560f * sphPos[2]),
                                    std::max(0.4f, 0.0556434f * sphPos[0] - 0.2040259f * sphPos[1] + 1.0572252f * sphPos[2]),
                
                            };
                    glColor3fv(col);
                }
                
                glVertex3fv(*v); // pointer method, more performant
                // glPopMatrix(); // RESTORES TO THE VIEW MATRIX
            }
        }
        glEnd();
        glDisable(GL_BLEND);
        glDisable(GL_POINT_SMOOTH);
    
    }
    else
    {
        std::cout << "\nPBO CALLED BUT NOT IMPLEMENTED YET";
    }
}


/// ARCHIVED COLOUR STUFF
// normalise mass and assign colour based upon it
// float normalM = (m_pos[4 * i + 3] -
//                  (float) INIT_M_HIGHER / ((float) INIT_M_HIGHER - (float) INIT_M_LOWER));
// GLfloat col[3] = {cosf(normalM + 0.5f * 40.0f), // red   -> high to begin with, ~0.17 at max mass
//                   sinf(normalM + 0.5f * 22.0f), // green -> zero to begin with, ~0.7 at max mass
//                   sinf(normalM + 0.5f * 45.0f)};  // blue  -> zero to begin with, 1 at max mass
// an attempt at RGB XYZ hue colours
// float normX = (m_pos[4*i] - 1000.0f())
// float normY = (m_pos[4*i+1] - ((float)HEIGHT/2) / (((float)HEIGHT/2) - (-1.f*(float)HEIGHT/2)));
// float normX = m_pos[4*i] - ((float)WIDTH/2) / ((float)WIDTH/2 - (-1.f*(float)WIDTH/2));
// float normY = m_pos[4*i+1] - ((float)WIDTH/2) / ((float)WIDTH/2 - (-1.f*(float)WIDTH/2));
// float normZ = (m_pos[4*i+2] - ((float)SYSTEM_THICKNESS) / (((float)SYSTEM_THICKNESS) - (-1.f*(float)SYSTEM_THICKNESS)));

// float phi = normX * ( 0.2f * (float)PI / 180.f );
// float theta = normY * ( 0.2f * (float)PI / 180.f );

// float r = sqrtf(normX * normX + normY * normY);
// float phi = ((float)PI * powf(r, 2) * (float)PI /180.0f);
// float theta = normY * ( 0.2f * (float)PI / 180.f );
// float theta = normY * ((float)PI * powf(r, PI) * (float)PI /180.0f);


// GLfloat  spherical(const float *pos)
// {
//     float phi = pos[0] * ( (float)PI / 180.f);
//     float theta = pos[1] * ( (float)PI / 180.f);
//
//     GLfloat sphPos[3];
//     sphPos[0] = cosf(phi) * cosf(theta);
//     sphPos[1] = sinf(phi) * cosf(theta);
//     sphPos[2] = sinf(theta);
//
//     return sphPos[0], sphPos[1], sphPos[2];
// }


///////////////////////////////
// -----ALL FROM NVIDIA----- //
///////////////////////////////
// const char vertexShader[] =     // FROM NVIDIA!
// {
//         "void main()                                                            \n"
//         "{                                                                      \n"
//         "    float pointSize = 500.0;                                           \n"
//         "    vec3 pos_eye = vec3 (gl_ModelViewMatrix * gl_Vertex);              \n"
//         "    gl_PointSize = max(1.0, pointSize / (1.0 - pos_eye.z));            \n"
//         "    gl_TexCoord[0] = gl_MultiTexCoord0;                                \n"
//         //"    gl_TexCoord[1] = gl_MultiTexCoord1;                                \n"
//         "    gl_Position = ftransform();                                        \n"
//         "    gl_FrontColor = gl_Color;                                          \n"
//         "}                                                                      \n"
// };
//
// const char pixelShader[] =      // FROM NVIDIA!
// {
//         "uniform sampler2D splatTexture;                                        \n"
//
//         "void main()                                                            \n"
//         "{                                                                      \n"
//         "    vec4 color = (0.6 + 0.4 * gl_Color) * texture2D(splatTexture, gl_TexCoord[0].st);           \n"
//         "    gl_FragColor = color * lerp(vec4(0.1, 0.0, 0.0, color.w), vec4(1.0, 0.7, 0.3, color.w), color.w);\n"
//         "}                                                                      \n"
// };
//
//
// void NbodyRenderer::_initGL()   // FROM NVIDIA!
// {
//     // // TODO: implement vertex shaders
//     m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
//     m_pixelShader = glCreateShader(GL_FRAGMENT_SHADER);
//
//     glutReportErrors();
//     const char* v = vertexShader;
//     const char* p = pixelShader;
//     glShaderSource(m_vertexShader, 1, &v, nullptr);
//     glShaderSource(m_pixelShader, 1, &p, nullptr);
//
//     glCompileShader(m_vertexShader);
//     glCompileShader(m_pixelShader);
//
//     m_program = glCreateProgram();
//
//     glAttachShader(m_program, m_vertexShader);
//     glAttachShader(m_program, m_pixelShader);
//
//     glLinkProgram(m_program);
//
//     _createTexture(32);
//
//     glGenBuffers(1, &m_vboColour); // TODO: assign vbo
//     glBindBuffer(GL_ARRAY_BUFFER_ARB, m_vboColour);
//     glBufferData(GL_ARRAY_BUFFER_ARB, N_BODIES * 4 * sizeof(float), nullptr, GL_STATIC_DRAW_ARB);
//     glBindBuffer(GL_ARRAY_BUFFER_ARB, 0);
//
// }
//
//
// //------------------------------------------------------------------------------
// // Function     	  : EvalHermite
// // Description	    :
// //------------------------------------------------------------------------------
// /**
// * EvalHermite(float pA, float pB, float vA, float vB, float u)
// * @brief Evaluates Hermite basis functions for the specified coefficients.
// */
// inline float evalHermite(float pA, float pB, float vA, float vB, float u)
// {
//     float u2=(u*u), u3=u2*u;
//     float B0 = 2*u3 - 3*u2 + 1;
//     float B1 = -2*u3 + 3*u2;
//     float B2 = u3 - 2*u2 + u;
//     float B3 = u3 - u;
//     return( B0*pA + B1*pB + B2*vA + B3*vB );
// }
//
//
// unsigned char* createGaussianMap(int N)
// {
//     auto *M = new float[2*N*N];
//     auto *B = new unsigned char[4*N*N];
//     float X,Y,Y2,Dist;
//     float Incr = 2.0f/N;
//     int i=0;
//     int j = 0;
//     Y = -1.0f;
//     //float mmax = 0;
//     for (int y=0; y<N; y++, Y+=Incr)
//     {
//         Y2=Y*Y;
//         X = -1.0f;
//         for (int x=0; x<N; x++, X+=Incr, i+=2, j+=4)
//         {
//             Dist = (float)sqrtf(X*X+Y2);
//             if (Dist>1) Dist=1;
//             M[i+1] = M[i] = evalHermite(1.0f,0,0,0,Dist);
//             B[j+3] = B[j+2] = B[j+1] = B[j] = (unsigned char)(M[i] * 255);
//         }
//     }
//     delete [] M;
//     return(B);
// }
//
// void NbodyRenderer::_createTexture(int resolution)
// {
//     unsigned char* data = createGaussianMap(resolution);
//     glGenTextures(1, &m_texture);
//     glBindTexture(GL_TEXTURE_2D, m_texture);
//     glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, resolution, resolution, 0,
//                  GL_RGBA, GL_UNSIGNED_BYTE, data);
//
// }






// OLD
// |> velocity based colours
// for (int i = 0; i < N_BODIES; i++) {
//     // colour based on velocity for now
//     float vxSqr = orbVel[i].x * orbVel[i].x;
//     float vySqr = orbVel[i].y * orbVel[i].y;
//     float vzSqr = orbVel[i].z * orbVel[i].z;
//     float vMag = sqrtf(vxSqr + vySqr + vzSqr);
//
//     // colour based on mass
//     float normalV = (vMag - (float)MAX_V) / ((float)MAX_V - (float)MIN_V);
//
//     float r = std::max<float>(sinf(normalV), 0.6f);
//     float g = std::max<float>(cosf(normalV + 0.5f), 0.6f);
//     float b = 1.2f - std::max<float>(sinf(normalV), 0.6f);
//
//     glColor3f(r, g, b);
//     glVertex3f(orbPos[i].x, orbPos[i].y, orbPos[i].z);
// }