//**************************************************
// \file ATLLArBarrelSensDet.cc
// \brief: implementation of ATLLArBarrelSensDet
//         class
// \author: Lorenzo Pezzotti (CERN EP-SFT-sim)
//          @lopezzot
// \start date: 21 february 2023
//**************************************************

// Includers from project files
//
#include "ATLLArBarrelSensDet.hh"

// Includers from Geant4
//
#include "G4HCofThisEvent.hh"
#include "G4Step.hh"
#include "G4ThreeVector.hh"
#include "G4SDManager.hh"
#include "G4ios.hh"

// Includers from STL
//
#include <cmath>

const G4String ATLLArBarrelSensDet::fFrontHitsCollectionName  = "FrontHitsCollectionName";
const G4String ATLLArBarrelSensDet::fBackHitsCollectionName   = "BackHitsCollectionName";
const G4String ATLLArBarrelSensDet::fMiddleHitsCollectionName = "MiddleHitsCollectionName";
const G4String ATLLArBarrelSensDet::fSensDetName              = "LArSD";

ATLLArBarrelSensDet::ATLLArBarrelSensDet(const G4String &name)
    : G4VSensitiveDetector(name), fFrontHitsCollection(nullptr), fMiddleHitsCollection(nullptr),
      fBackHitsCollection(nullptr), fFrontHCID(-1), fMiddleHCID(-1), fBackHCID(-1)
{

  // Insert in vector of collection names the 3 collections' names
  collectionName.insert(fFrontHitsCollectionName);
  collectionName.insert(fMiddleHitsCollectionName);
  collectionName.insert(fBackHitsCollectionName);
}

void ATLLArBarrelSensDet::Initialize(G4HCofThisEvent *HCE)
{

  /*This method is called at the beginning of each event.
    The user must allocate with "new" the hits collections of this SD
    and link them to the SD field pointers.
    The user should as well add them to the hit collection of this event
    in such a way hit collections are automatically cleaned at the end
    of each event and can be accessed in the EndOfEvent method with
    evet->GetHCofThisEvent().*/

  // Hits Collections must have a unique pair of string (SDname + hitscollectionname)
  //
  fFrontHitsCollection  = new ATLLArBarrelHitsCollection(GetName(), fFrontHitsCollectionName);
  fMiddleHitsCollection = new ATLLArBarrelHitsCollection(GetName(), fMiddleHitsCollectionName);
  fBackHitsCollection   = new ATLLArBarrelHitsCollection(GetName(), fBackHitsCollectionName);

  // Hits Collections (among all SD of the simulations) are also identified by an integer.
  // It can be accessed via the SDManager anytime for both storing and retrieving the hits collection.
  // If the hitcollection name is unambigous, it is enough to retrieve the ID (no need for SD name).
  //
  if (fFrontHCID < 0) { // prevent doing it every event -> HCIDs are identical for every event
    fFrontHCID  = G4SDManager::GetSDMpointer()->GetCollectionID(fFrontHitsCollectionName);
    fMiddleHCID = G4SDManager::GetSDMpointer()->GetCollectionID(fMiddleHitsCollectionName);
    fBackHCID   = G4SDManager::GetSDMpointer()->GetCollectionID(fBackHitsCollectionName);
  }
  HCE->AddHitsCollection(fFrontHCID, fFrontHitsCollection);
  HCE->AddHitsCollection(fMiddleHCID, fMiddleHitsCollection);
  HCE->AddHitsCollection(fBackHCID, fBackHitsCollection);

  // Reset the per-event boundary diagnostics
  //
  fEdepCrossCell    = 0.;
  fEdepCrossSection = 0.;

  // Allocate fixed number of hits
  //
  for (std::size_t i = 0; i < ATLLArBarrelTBConstants::FrontHitNo; i++) {
    fFrontHitsCollection->insert(new ATLLArBarrelHit());
  }
  for (std::size_t i = 0; i < ATLLArBarrelTBConstants::MiddleHitNo; i++) {
    fMiddleHitsCollection->insert(new ATLLArBarrelHit());
  }
  for (std::size_t i = 0; i < ATLLArBarrelTBConstants::BackHitNo; i++) {
    fBackHitsCollection->insert(new ATLLArBarrelHit());
  }
}

G4bool ATLLArBarrelSensDet::ProcessHits(G4Step *aStep, G4TouchableHistory *th)
{

  // Print out some info step-by-step in sensitive elements
  //
  // G4cout<<"Track #: "<< aStep->GetTrack()->GetTrackID()<< " " <<
  //         "Step #: " << aStep->GetTrack()->GetCurrentStepNumber()<< " "<<
  //         "Volume: " << aStep->GetPreStepPoint()->GetTouchableHandle()->GetVolume()->GetName()<< " " << G4endl;
  // G4cout<<"x: "<< aStep->GetPreStepPoint()->GetPosition().x() <<
  //         "y: "<< aStep->GetPreStepPoint()->GetPosition().y() <<
  //         "z: "<< aStep->GetPreStepPoint()->GetPosition().z() << G4endl;
  // G4cout<<"Particle "<< aStep->GetTrack()->GetParticleDefinition()->GetParticleName()<< " " <<
  //         "Dep(MeV) "<< aStep->GetTotalEnergyDeposit() << " " <<
  //         "Mat "     << aStep->GetPreStepPoint()->GetMaterial()->GetName() << " " << G4endl;

  // Find the readout cell of the pre-step point. Steps with no deposit or outside the readout
  // (eta < 0, eta >= 0.8, i.e. only the A side, or phi outside the module, because the
  // DeltaPhi of the single module does not contain the accordion tips entirely) are skipped
  //
  auto Edep = aStep->GetTotalEnergyDeposit();
  if (Edep == 0.) return false;
  const Cell pre = FindCell(aStep->GetPreStepPoint()->GetPosition());
  if (!pre.inside) return false;
  const G4double EdepNoBirks = Edep;
  Edep                       = ApplyBirks(Edep, aStep->GetStepLength() / 10.);

  const auto Section  = pre.section;
  const G4int EtaIdx  = pre.etaIdx;
  const G4int PhiIdx  = pre.phiIdx;
  const G4int HitID   = pre.hitID;
  const auto DeltaEta = pre.deltaEta;
  const auto DeltaPhi = pre.deltaPhi;
  CheckHitID(Section, HitID); // throws fatal exception if >=MaxHitNo

  // Boundary diagnostics: the whole deposit goes to the pre-step cell, also when the step ends
  // in another cell or section
  //
  const Cell post = FindCell(aStep->GetPostStepPoint()->GetPosition());
  if (!post.inside || post.section != Section || post.hitID != HitID) fEdepCrossCell += Edep;
  if (post.inside && post.section != Section) fEdepCrossSection += Edep;

  // Get hit according to section and HitID
  //
  ATLLArBarrelHit *hit = nullptr;
  switch (Section) {
  case ATLLArBarrelTBConstants::STACSection::Front:
    hit = (*fFrontHitsCollection)[HitID];
    break;
  case ATLLArBarrelTBConstants::STACSection::Middle:
    hit = (*fMiddleHitsCollection)[HitID];
    break;
  case ATLLArBarrelTBConstants::STACSection::Back:
    hit = (*fBackHitsCollection)[HitID];
    break;
  default:
    G4ExceptionDescription msg;
    msg << "Wrong enum section value ";
    G4Exception("CalorimeterSD::ProcessHits()", "MyCode0004", FatalException, msg);
  }
  if (!hit) {
    G4ExceptionDescription msg;
    msg << "Cannot retrieve hit with HitID " << HitID;
    G4Exception("CalorimeterSD::ProcessHits()", "MyCode0004", FatalException, msg);
  }

  // Add edep to hit
  //
  hit->AddEdep(Edep);
  hit->AddEdepNoBirks(EdepNoBirks);
  if (!hit->HasPositionBeenAllocated()) {
    hit->SetEta(EtaIdx * DeltaEta + DeltaEta / 2.);
    hit->SetPhi(PhiIdx * DeltaPhi + DeltaPhi / 2.);
    hit->SetPositionAllocated(true);
  }

  return true;
}

ATLLArBarrelSensDet::Cell ATLLArBarrelSensDet::FindCell(const G4ThreeVector &position) const
{

  Cell cell;
  const auto Eta   = position.eta();
  const auto Phi   = position.phi();   // rad
  const auto Theta = position.theta(); // rad
  if (Eta < ATLLArBarrelTBConstants::EtaMin || Eta >= ATLLArBarrelTBConstants::EtaChange) return cell;
  if (Phi >= ATLLArBarrelTBConstants::halfSTACDeltaPhi || Phi < (-ATLLArBarrelTBConstants::halfSTACDeltaPhi))
    return cell;

  // Depth inside STAC
  //
  const G4double Depth = position.mag() - (ATLLArBarrelTBConstants::RMinSTAC / std::sin(Theta));
  G4double EtasPerRow  = 0.;
  if (Depth < ATLLArBarrelTBConstants::FrontDepth) {
    cell.section  = ATLLArBarrelTBConstants::STACSection::Front;
    cell.deltaEta = ATLLArBarrelTBConstants::FrontDeltaEta;
    cell.deltaPhi = ATLLArBarrelTBConstants::FrontDeltaPhi;
    EtasPerRow    = ATLLArBarrelTBConstants::FrontEtasPerRow;
  } else if (Depth < ATLLArBarrelTBConstants::MiddleDepth) {
    cell.section  = ATLLArBarrelTBConstants::STACSection::Middle;
    cell.deltaEta = ATLLArBarrelTBConstants::MiddleDeltaEta;
    cell.deltaPhi = ATLLArBarrelTBConstants::MiddleDeltaPhi;
    EtasPerRow    = ATLLArBarrelTBConstants::MiddleEtasPerRow;
  } else {
    cell.section  = ATLLArBarrelTBConstants::STACSection::Back;
    cell.deltaEta = ATLLArBarrelTBConstants::BackDeltaEta;
    cell.deltaPhi = ATLLArBarrelTBConstants::BackDeltaPhi;
    EtasPerRow    = ATLLArBarrelTBConstants::BackEtasPerRow;
  }

  // Eta and Phi index (max EtaIdx value = 256, max PhiIdx value = 16) and index in the collection
  //
  cell.etaIdx = std::floor(Eta / cell.deltaEta);
  cell.phiIdx = std::floor((Phi + ATLLArBarrelTBConstants::halfSTACDeltaPhi) / cell.deltaPhi);
  cell.hitID  = cell.phiIdx * EtasPerRow + cell.etaIdx;
  cell.inside = true;
  return cell;
}

void ATLLArBarrelSensDet::EndOfEvent(G4HCofThisEvent *) {}

//**************************************************
