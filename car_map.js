let map;
let directionsService;
let directionsRenderer;

function initMap() {
  map = new google.maps.Map(document.getElementById("map"), {
    center: { lat: 22.319353103637695, lng: 114.16929626464844 }, // Default to New York City
    zoom: 12,
  });
  directionsService = new google.maps.DirectionsService();
  directionsRenderer = new google.maps.DirectionsRenderer();
  directionsRenderer.setMap(map);
}

const trafficJams = [
    { lat: 40.730610, lng: -73.935242, severity: 'high', duration: '30 minutes' }, // Example traffic jam in New York
    // Add more traffic jam data as needed
];

function calculateAndDisplayRoute() {
  const start = document.getElementById("start").value;
  const end = document.getElementById("end").value;

  //const waypoints = [{location: document.getElementById("stopover").value, stopover: true}];

  const request = {
    origin: start,
    destination: end,
    //waypoints: waypoints,
    travelMode: google.maps.TravelMode.DRIVING,
  };

  directionsService.route(request, (result, status) => {
    if (status == google.maps.DirectionsStatus.OK) {
      directionsRenderer.setDirections(result);
    } else {
      window.alert("Directions request failed due to " + status);
    }
  });
}
