angular.module('bdbServices', ['ngResource'])
.factory('Table', function($resource) {
        return $resource('/api/v1/table/:tableName', {}, {
            query: {method:'GET', params:{tableName:''}, isArray:true}
        });
    });


angular.module('bdbRestExampleApp', ['ngResource', 'bdbServices'])
    .controller('TablesListCtrl', function($scope, Table) {
        $scope.tables = Table.query();
    });
