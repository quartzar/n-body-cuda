//
// Created by quartzar on 21/10/22.
//

#ifndef ORBITERV6_NBODYRENDERER_H
#define ORBITERV6_NBODYRENDERER_H


class NbodyRenderer {

public:
    NbodyRenderer();
    ~NbodyRenderer();
    
    void setPositions(float *pos);
    void setVelocities(float *vel);
    
    enum RenderMode
    {
        POINTS,
        SPRITES
    };
    
    void display(RenderMode mode, float zoom, float xRot, float yRot, float zRot,
                 float xTrans, float yTrans, float zTrans, bool trailMode, bool colourMode);
    
    
protected: // METHODS
    void _initGL();
    void _createTexture(int resolution);
    void _drawOrbitals(bool color);
    
    // struct Colour { float r, g, b; };
    
protected: // DATA
    float *m_pos;
    float *m_vel;
    
    uint m_program;
    uint m_texture;
    uint m_vertexShader;
    uint m_pixelShader;
    uint m_pbo;
    uint m_vboColour;
    
};


#endif //ORBITERV6_NBODYRENDERER_H
