SetFactory("OpenCASCADE");

Mesh.CharacteristicLengthMin = 5.0;
Mesh.CharacteristicLengthMax = 5.0;
Mesh.MshFileVersion = 2.2;

Merge "1_dot.brep";

//+
Recursive Delete {
  Volume{16}; Volume{14}; Volume{15}; Volume{12}; Volume{13}; Volume{8}; Volume{7}; 
}
//+
Recursive Delete {
  Volume{11}; Volume{10}; Volume{9}; 
}
//+
Physical Volume("SiGe_1", 151) = {1};
//+
Physical Volume("SiGe_2", 152) = {5};
//+
Physical Volume("Al2O3", 153) = {6};
//+
Physical Volume("SiGe_1_dot", 154) = {2};
//+
Physical Volume("SiGe_2_dot", 155) = {4};
//+
Physical Volume("Si", 156) = {3};
//+
Physical Surface("QS0", 157) = {29};
//+
Physical Surface("QS1", 158) = {33};
//+
Physical Surface("QP0", 159) = {49};
//+
Physical Surface("QT0", 160) = {48};
//+
Physical Surface("QT1", 161) = {50};

// -----------------------------------------------------------------------------
// Ohmic/back-contact boundary for QTCAD
// -----------------------------------------------------------------------------
// This labels only the bottom face of the stack (z = 0 nm in the BREP).
// It does NOT label the Si quantum-well sidewalls, avoiding a too-small
// Schrödinger eigenproblem on the dot submesh.
eps = 1e-3;
Si_ohmic_bnd[] = Surface In BoundingBox {-75-eps, -75-eps, -eps, 75+eps, 75+eps, eps};
Physical Surface("Si_ohmic", 162) = {Si_ohmic_bnd[]};

// Mesh the structure
Mesh 1;
Mesh 2;
Mesh 3;

// Save the mesh
Save "1_dot.msh";

// Save the geometry
Save "1_dot.xao";

